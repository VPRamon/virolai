//! MAPPO (Multi-Agent PPO) trainer (§9).
//!
//! Implements Centralized Training, Decentralized Execution (CTDE):
//! - Shared actor network uses local observations.
//! - Centralized critic uses global state.

use std::path::Path;

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use rand::RngExt;

use super::buffer::{RolloutBuffer, Transition};
use super::gae::compute_gae;
use crate::algorithms::rl::agent::AgentState;
use crate::algorithms::rl::config::RLConfig;
use crate::algorithms::rl::environment::RLEnvironment;
use crate::algorithms::rl::metrics::EvaluationMetrics;
use crate::algorithms::rl::network::{ActorNetwork, CriticNetwork, NeuralPolicy};
use crate::algorithms::rl::observation::ObservationBuilder;
use crate::algorithms::rl::types::AgentType;

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
    /// Evaluate policy every N updates (0 = disabled).
    pub eval_interval: u32,
    /// Number of episodes for evaluation.
    pub eval_episodes: usize,
    /// Save checkpoint every N updates (0 = disabled).
    pub checkpoint_interval: u32,
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
            eval_interval: 50,
            eval_episodes: 10,
            checkpoint_interval: 100,
        }
    }
}

/// Clips gradient norms for all trainable variables in a `VarStore`.
///
/// Implements the same logic as PyTorch's `torch.nn.utils.clip_grad_norm_`:
/// computes the total L2 norm of all gradients, and scales them down if
/// the norm exceeds `max_norm`.
fn clip_grad_norm(vs: &nn::VarStore, max_norm: f64) {
    let vars = vs.trainable_variables();
    let total_norm_sq: f64 = vars
        .iter()
        .map(|v| {
            let g = v.grad();
            if g.defined() {
                g.pow_tensor_scalar(2)
                    .sum(Kind::Float)
                    .double_value(&[])
            } else {
                0.0
            }
        })
        .sum();
    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for var in &vars {
            let g = var.grad();
            if g.defined() {
                let _ = g.multiply_scalar_(clip_coef);
            }
        }
    }
}

/// Derives agent composition `(count, AgentType)` pairs from a slice of agents.
///
/// Used to create a separate evaluation environment with the same agent
/// makeup as the training environment, without sharing RNG state.
fn derive_agent_composition(agents: &[AgentState]) -> Vec<(u32, AgentType)> {
    let mut young = 0u32;
    let mut middle = 0u32;
    let mut old = 0u32;
    for agent in agents {
        match agent.agent_type {
            AgentType::Young => young += 1,
            AgentType::Middle => middle += 1,
            AgentType::Old => old += 1,
        }
    }
    let mut composition = Vec::new();
    if young > 0 {
        composition.push((young, AgentType::Young));
    }
    if middle > 0 {
        composition.push((middle, AgentType::Middle));
    }
    if old > 0 {
        composition.push((old, AgentType::Old));
    }
    composition
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

            // Periodic evaluation (uses a separate env to avoid perturbing training RNG)
            if self.train_config.eval_interval > 0
                && update % self.train_config.eval_interval == 0
            {
                let mut eval_policy =
                    NeuralPolicy::from_actor_var_store(self.actor.var_store(), &self.env_config);
                eval_policy.set_greedy(true);

                // Build agent composition from the training env
                let composition = derive_agent_composition(&env.agents);
                let mut eval_env = RLEnvironment::new(self.env_config.clone(), 1_000_000);
                eval_env.set_agents(&composition);

                let metrics = EvaluationMetrics::evaluate(
                    &mut eval_env,
                    &mut eval_policy,
                    self.train_config.eval_episodes,
                );
                eprintln!("[Eval @ update {}]\n{}", update, metrics);
            }

            // Periodic checkpoint
            if self.train_config.checkpoint_interval > 0
                && update > 0
                && update % self.train_config.checkpoint_interval == 0
            {
                let dir = format!("checkpoints/update_{}", update);
                if let Err(e) = self.save_checkpoint(Path::new(&dir)) {
                    eprintln!("[Warning] Checkpoint save failed: {}", e);
                } else {
                    eprintln!("[Checkpoint] Saved to {}", dir);
                }
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

    /// Performs PPO policy and value updates with minibatch sampling and
    /// gradient clipping.
    ///
    /// Returns `(mean_actor_loss, mean_critic_loss)`.
    fn ppo_update(&mut self, buffer: &RolloutBuffer) -> (f64, f64) {
        let n = buffer.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let batch_size = self.train_config.batch_size.min(n);
        let mut total_actor_loss = 0.0;
        let mut total_critic_loss = 0.0;
        let mut n_updates = 0;

        for _ in 0..self.train_config.n_epochs {
            // Shuffle indices for this epoch (Fisher-Yates)
            let mut indices: Vec<usize> = (0..n).collect();
            let mut rng = rand::rng();
            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            // Process minibatches
            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let batch_indices = &indices[batch_start..batch_end];

                let mut batch_actor_loss = Tensor::zeros([], (Kind::Float, self.device));
                let mut batch_critic_loss = Tensor::zeros([], (Kind::Float, self.device));
                let mut batch_count = 0;

                for &idx in batch_indices {
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
                    let actions_tensor = Tensor::from_slice(
                        &t.actions.iter().map(|&a| a as i64).collect::<Vec<_>>(),
                    )
                    .to_kind(Kind::Int64);
                    let old_log_probs_tensor =
                        Tensor::from_slice(&t.log_probs).to_kind(Kind::Float);

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

                    batch_actor_loss = batch_actor_loss + &actor_total;

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

                    batch_critic_loss = batch_critic_loss + &critic_loss;
                    batch_count += 1;

                    total_actor_loss += f64::try_from(&actor_loss).unwrap_or(0.0);
                    total_critic_loss += f64::try_from(&critic_loss).unwrap_or(0.0);
                }

                if batch_count > 0 {
                    // Average over minibatch
                    let mean_actor = &batch_actor_loss / batch_count as f64;
                    let mean_critic = &batch_critic_loss / batch_count as f64;

                    // Actor backward + clip + step
                    self.actor_opt.zero_grad();
                    mean_actor.backward();
                    clip_grad_norm(self.actor.var_store(), self.train_config.max_grad_norm);
                    self.actor_opt.step();

                    // Critic backward + clip + step
                    self.critic_opt.zero_grad();
                    mean_critic.backward();
                    clip_grad_norm(
                        self.critic.var_store(),
                        self.train_config.max_grad_norm,
                    );
                    self.critic_opt.step();

                    n_updates += batch_count;
                }
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

    /// Saves actor and critic checkpoints to `dir`.
    ///
    /// Creates the directory if it does not exist. Saves:
    /// - `dir/actor.pt` — actor network weights
    /// - `dir/critic.pt` — critic network weights
    pub fn save_checkpoint(&self, dir: &Path) -> Result<(), tch::TchError> {
        std::fs::create_dir_all(dir).map_err(|e| {
            tch::TchError::FileFormat(format!("Failed to create checkpoint dir: {}", e))
        })?;
        self.actor.var_store().save(dir.join("actor.pt"))?;
        self.critic.var_store().save(dir.join("critic.pt"))?;
        Ok(())
    }

    /// Loads actor and critic weights from a checkpoint directory.
    ///
    /// Expects `dir/actor.pt` and `dir/critic.pt` to exist.
    pub fn load_checkpoint(&mut self, dir: &Path) -> Result<(), tch::TchError> {
        self.actor.var_store_mut().load(dir.join("actor.pt"))?;
        self.critic.var_store_mut().load(dir.join("critic.pt"))?;
        Ok(())
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

    #[test]
    fn train_with_minibatches() {
        // Ensure batch_size < total samples triggers minibatch splitting
        let env_config = RLConfig {
            episode_horizon: 10,
            ..RLConfig::default()
        };
        let train_config = TrainingConfig {
            n_episodes_per_update: 2,
            n_epochs: 2,
            batch_size: 4, // Small batches to force multiple minibatches
            ..TrainingConfig::default()
        };
        let mut env = RLEnvironment::new(env_config.clone(), 42);
        env.set_agents(&[(1, AgentType::Young), (1, AgentType::Old)]);
        let mut trainer = MAPPOTrainer::new(env_config, train_config, 2, Device::Cpu);

        let curve = trainer.train(&mut env, 2);
        assert_eq!(curve.len(), 2);
        // Both episodes should produce finite rewards
        for r in &curve {
            assert!(r.is_finite(), "Reward should be finite, got {}", r);
        }
    }

    #[test]
    fn gradient_clipping_limits_norm() {
        // Create a small tensor, set a large gradient, verify clipping
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let _w = root.zeros("w", &[10]);

        // Manually set gradients to large values
        for (_, mut var) in vs.variables() {
            let grad = tch::Tensor::full(var.size().as_slice(), 100.0, (tch::Kind::Float, Device::Cpu));
            var.set_grad(&grad);
        }

        clip_grad_norm(&vs, 1.0);

        // After clipping, the gradient norm should be close to max_norm (1.0)
        let mut total_norm_sq = 0.0f64;
        for (_, var) in vs.variables() {
            let g = var.grad();
            total_norm_sq += f64::try_from(g.pow_tensor_scalar(2).sum(tch::Kind::Double)).unwrap();
        }
        let total_norm = total_norm_sq.sqrt();
        assert!(
            total_norm < 1.0 + 0.01,
            "Clipped norm should be <= 1.0, got {}",
            total_norm
        );
    }

    #[test]
    fn checkpoint_save_load_roundtrip() {
        let env_config = RLConfig {
            episode_horizon: 5,
            ..RLConfig::default()
        };
        let train_config = TrainingConfig::default();
        let mut trainer = MAPPOTrainer::new(env_config.clone(), train_config.clone(), 2, Device::Cpu);

        let dir = std::env::temp_dir().join("virolai_test_checkpoint");
        let _ = std::fs::create_dir_all(&dir);

        // Save checkpoint
        trainer.save_checkpoint(&dir).expect("save should succeed");

        // Verify files exist
        assert!(dir.join("actor.pt").exists());
        assert!(dir.join("critic.pt").exists());

        // Load into new trainer
        let mut trainer2 = MAPPOTrainer::new(env_config, train_config, 2, Device::Cpu);
        trainer2
            .load_checkpoint(&dir)
            .expect("load should succeed");

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
    }
}
