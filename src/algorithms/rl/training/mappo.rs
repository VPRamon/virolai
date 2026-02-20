//! MAPPO (Multi-Agent PPO) trainer (§9).
//!
//! Implements Centralized Training, Decentralized Execution (CTDE):
//! - Shared actor network uses local observations.
//! - Centralized critic uses global state.

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use super::buffer::{RolloutBuffer, Transition};
use super::gae::compute_gae;
use crate::algorithms::rl::config::RLConfig;
use crate::algorithms::rl::environment::RLEnvironment;
use crate::algorithms::rl::network::{ActorNetwork, CriticNetwork};
use crate::algorithms::rl::observation::ObservationBuilder;

/// Training hyperparameters for MAPPO.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Discount factor γ.
    pub gamma: f64,
    /// GAE λ parameter.
    pub gae_lambda: f64,
    /// PPO clip parameter ε.
    pub clip_eps: f64,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Value loss coefficient.
    pub value_coef: f64,
    /// Actor learning rate.
    pub lr_actor: f64,
    /// Critic learning rate.
    pub lr_critic: f64,
    /// Number of PPO optimization epochs per update.
    pub n_epochs: u32,
    /// Mini-batch size for PPO updates.
    pub batch_size: usize,
    /// Number of episodes to collect per update.
    pub n_episodes_per_update: u32,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            lr_actor: 3e-4,
            lr_critic: 1e-3,
            n_epochs: 4,
            batch_size: 64,
            n_episodes_per_update: 4,
            max_grad_norm: 0.5,
        }
    }
}

/// MAPPO trainer implementing Centralized Training, Decentralized Execution.
///
/// Trains a shared actor network (one for all agents) using PPO with a
/// centralized critic. The actor uses per-agent local observations while
/// the critic sees the global state.
pub struct MAPPOTrainer {
    /// Shared actor network.
    pub actor: ActorNetwork,
    /// Centralized critic network.
    pub critic: CriticNetwork,
    /// Training hyperparameters.
    pub train_config: TrainingConfig,
    /// Environment configuration.
    pub env_config: RLConfig,
    /// Actor optimizer.
    actor_opt: nn::Optimizer,
    /// Critic optimizer.
    critic_opt: nn::Optimizer,
    /// Device (CPU/CUDA).
    device: Device,
}

impl MAPPOTrainer {
    /// Creates a new MAPPO trainer.
    ///
    /// # Arguments
    ///
    /// * `env_config` - Environment configuration
    /// * `train_config` - Training hyperparameters
    /// * `n_agents` - Number of agents (needed for global state dim)
    /// * `device` - Compute device
    pub fn new(
        env_config: RLConfig,
        train_config: TrainingConfig,
        n_agents: usize,
        device: Device,
    ) -> Self {
        let obs_dim = env_config.observation_dim();
        let action_dim = env_config.action_dim();
        let global_state_dim = ObservationBuilder::global_state_dim(n_agents, &env_config);

        let mut actor = ActorNetwork::new(obs_dim, action_dim, device);
        let critic = CriticNetwork::new(global_state_dim, device);

        let actor_opt = nn::Adam::default()
            .build(actor.var_store_mut(), train_config.lr_actor)
            .expect("Failed to create actor optimizer");
        let critic_opt = nn::Adam::default()
            .build(critic.var_store_mut(), train_config.lr_critic)
            .expect("Failed to create critic optimizer");

        Self {
            actor,
            critic,
            train_config,
            env_config,
            actor_opt,
            critic_opt,
            device,
        }
    }

    /// Runs the full MAPPO training loop.
    ///
    /// # Arguments
    ///
    /// * `env` - The RL environment
    /// * `total_updates` - Number of policy update iterations
    ///
    /// # Returns
    ///
    /// A vector of (update_index, mean_episode_reward) for the learning curve.
    pub fn train(&mut self, env: &mut RLEnvironment, total_updates: u32) -> Vec<(u32, f64)> {
        let mut learning_curve = Vec::new();

        for update in 0..total_updates {
            // Collect rollouts
            let mut buffer = RolloutBuffer::new();
            let mut episode_rewards = Vec::new();

            for _ in 0..self.train_config.n_episodes_per_update {
                let episode_reward = self.collect_rollout(env, &mut buffer);
                episode_rewards.push(episode_reward);
            }

            // Compute GAE
            let rewards: Vec<f64> = buffer.transitions.iter().map(|t| t.reward).collect();
            let values: Vec<f64> = buffer.transitions.iter().map(|t| t.value).collect();
            let dones: Vec<bool> = buffer.transitions.iter().map(|t| t.done).collect();

            let (advantages, returns) = compute_gae(
                &rewards,
                &values,
                &dones,
                self.train_config.gamma,
                self.train_config.gae_lambda,
            );
            buffer.set_advantages_and_returns(advantages, returns);
            buffer.normalize_advantages();

            // PPO update
            let (actor_loss, critic_loss) = self.ppo_update(&buffer);

            let mean_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
            learning_curve.push((update, mean_reward));

            if update % 10 == 0 {
                eprintln!(
                    "[Update {}/{}] mean_reward={:.3} actor_loss={:.4} critic_loss={:.4}",
                    update, total_updates, mean_reward, actor_loss, critic_loss
                );
            }
        }

        learning_curve
    }

    /// Collects one episode of rollout data.
    fn collect_rollout(&self, env: &mut RLEnvironment, buffer: &mut RolloutBuffer) -> f64 {
        let mut obs = env.reset();

        loop {
            // Get global state for critic
            let global_state = ObservationBuilder::build_global_state(
                &env.agents,
                &env.task_pool,
                &self.env_config,
            );

            // Compute value estimate
            let state_tensor = Tensor::from_slice(&global_state)
                .unsqueeze(0)
                .to_kind(Kind::Float);
            let value: f64 = self.critic.forward(&state_tensor).double_value(&[0]);

            // Sample actions from actor
            let n_agents = obs.len();
            let obs_dim = if n_agents > 0 { obs[0].len() } else { 0 };
            let flat_obs: Vec<f64> = obs.iter().flat_map(|o| o.iter().copied()).collect();

            let (actions_vec, log_probs_vec) = if n_agents > 0 {
                let obs_tensor = Tensor::from_slice(&flat_obs)
                    .reshape([n_agents as i64, obs_dim as i64])
                    .to_kind(Kind::Float);
                let (actions_t, log_probs_t) = self.actor.sample_actions(&obs_tensor);
                let actions: Vec<i64> = actions_t.into();
                let log_probs: Vec<f64> = log_probs_t.into();
                (
                    actions.iter().map(|&a| a as usize).collect::<Vec<_>>(),
                    log_probs,
                )
            } else {
                (vec![], vec![])
            };

            // Step environment
            let result = env.step(actions_vec.clone());

            // Store transition
            buffer.add(Transition {
                observations: obs.clone(),
                actions: actions_vec,
                reward: result.reward,
                log_probs: log_probs_vec,
                value,
                done: result.done,
                global_state,
            });

            obs = result.observations;

            if result.done {
                break;
            }
        }

        env.cumulative_reward
    }

    /// Performs PPO policy and value updates.
    ///
    /// Returns `(mean_actor_loss, mean_critic_loss)`.
    fn ppo_update(&mut self, buffer: &RolloutBuffer) -> (f64, f64) {
        let n = buffer.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut total_actor_loss = 0.0;
        let mut total_critic_loss = 0.0;
        let mut n_updates = 0;

        for _ in 0..self.train_config.n_epochs {
            // Process all transitions (simplified: full batch for now)
            for idx in 0..n {
                let t = &buffer.transitions[idx];
                let advantage = buffer.advantages[idx];
                let ret = buffer.returns[idx];

                let n_agents = t.observations.len();
                if n_agents == 0 {
                    continue;
                }
                let obs_dim = t.observations[0].len();

                // Actor loss (sum over agents)
                let flat_obs: Vec<f64> = t
                    .observations
                    .iter()
                    .flat_map(|o| o.iter().copied())
                    .collect();
                let obs_tensor = Tensor::from_slice(&flat_obs)
                    .reshape([n_agents as i64, obs_dim as i64])
                    .to_kind(Kind::Float);
                let actions_tensor =
                    Tensor::from_slice(&t.actions.iter().map(|&a| a as i64).collect::<Vec<_>>())
                        .to_kind(Kind::Int64);
                let old_log_probs_tensor = Tensor::from_slice(&t.log_probs).to_kind(Kind::Float);

                let new_log_probs = self.actor.log_prob(&obs_tensor, &actions_tensor);
                let ratio = (&new_log_probs - &old_log_probs_tensor).exp();

                let adv_tensor =
                    Tensor::from_slice(&vec![advantage as f32; n_agents]).to_kind(Kind::Float);

                let surr1 = &ratio * &adv_tensor;
                let surr2 = ratio.clamp(
                    1.0 - self.train_config.clip_eps,
                    1.0 + self.train_config.clip_eps,
                ) * &adv_tensor;
                let actor_loss = -surr1.min_other(&surr2).mean(Kind::Float);

                // Entropy bonus
                let entropy = self.actor.entropy(&obs_tensor).mean(Kind::Float);
                let actor_total = &actor_loss - self.train_config.entropy_coef * &entropy;

                // Actor backward
                self.actor_opt.zero_grad();
                actor_total.backward();
                self.actor_opt.step();

                // Critic loss
                let state_tensor = Tensor::from_slice(&t.global_state)
                    .unsqueeze(0)
                    .to_kind(Kind::Float);
                let value_pred = self.critic.forward(&state_tensor);
                let ret_tensor = Tensor::from_slice(&[ret as f32]).to_kind(Kind::Float);
                let critic_loss = (&value_pred - &ret_tensor)
                    .pow_tensor_scalar(2)
                    .mean(Kind::Float)
                    * self.train_config.value_coef;

                self.critic_opt.zero_grad();
                critic_loss.backward();
                self.critic_opt.step();

                total_actor_loss += f64::try_from(&actor_loss).unwrap_or(0.0);
                total_critic_loss += f64::try_from(&critic_loss).unwrap_or(0.0);
                n_updates += 1;
            }
        }

        if n_updates > 0 {
            (
                total_actor_loss / n_updates as f64,
                total_critic_loss / n_updates as f64,
            )
        } else {
            (0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rl::{AgentType, RLConfig, RLEnvironment};

    #[test]
    fn trainer_creation() {
        let env_config = RLConfig::default();
        let train_config = TrainingConfig::default();
        let _trainer = MAPPOTrainer::new(env_config, train_config, 6, Device::Cpu);
    }

    #[test]
    fn train_smoke_test() {
        let env_config = RLConfig {
            episode_horizon: 5,
            ..RLConfig::default()
        };
        let train_config = TrainingConfig {
            n_episodes_per_update: 1,
            n_epochs: 1,
            ..TrainingConfig::default()
        };
        let mut env = RLEnvironment::new(env_config.clone(), 42);
        env.set_agents(&[(1, AgentType::Young), (1, AgentType::Old)]);
        let mut trainer = MAPPOTrainer::new(env_config, train_config, 2, Device::Cpu);

        let curve = trainer.train(&mut env, 2);
        assert_eq!(curve.len(), 2);
    }
}
