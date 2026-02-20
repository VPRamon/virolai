//! Neural network policy using tch-rs (PyTorch bindings).
//!
//! Provides MLP-based actor and critic networks for MAPPO training.
//! This module is only available with the `rl-nn` feature.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use super::config::RLConfig;
use super::policy::Policy;

/// MLP actor network that outputs action probabilities.
///
/// Architecture: `obs_dim → 128 → 64 → action_dim` with ReLU activations
/// and softmax output.
pub struct ActorNetwork {
    vs: nn::VarStore,
    net: nn::Sequential,
    action_dim: usize,
}

impl ActorNetwork {
    /// Creates a new actor network.
    pub fn new(obs_dim: usize, action_dim: usize, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let net = nn::seq()
            .add(nn::linear(
                p / "l1",
                obs_dim as i64,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(p / "l2", 128, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                p / "l3",
                64,
                action_dim as i64,
                Default::default(),
            ));

        Self {
            vs,
            net,
            action_dim,
        }
    }

    /// Forward pass: returns log-probabilities over actions.
    pub fn forward(&self, obs: &Tensor) -> Tensor {
        let logits = self.net.forward(obs);
        logits.log_softmax(-1, Kind::Float)
    }

    /// Samples actions from the policy distribution.
    pub fn sample_actions(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let log_probs = self.forward(obs);
        let probs = log_probs.exp();
        let actions = probs.multinomial(1, true).squeeze_dim(-1);
        let selected_log_probs = log_probs
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1);
        (actions, selected_log_probs)
    }

    /// Returns log-probabilities for given actions.
    pub fn log_prob(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let log_probs = self.forward(obs);
        log_probs
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    /// Returns the entropy of the policy distribution.
    pub fn entropy(&self, obs: &Tensor) -> Tensor {
        let log_probs = self.forward(obs);
        let probs = log_probs.exp();
        -(probs * log_probs).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
    }

    /// Returns a mutable reference to the variable store for optimization.
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Returns a reference to the variable store.
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
}

/// MLP critic network (centralized value function).
///
/// Architecture: `global_state_dim → 128 → 64 → 1` with ReLU activations.
pub struct CriticNetwork {
    vs: nn::VarStore,
    net: nn::Sequential,
}

impl CriticNetwork {
    /// Creates a new critic network.
    pub fn new(state_dim: usize, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let net = nn::seq()
            .add(nn::linear(
                p / "l1",
                state_dim as i64,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(p / "l2", 128, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(p / "l3", 64, 1, Default::default()));

        Self { vs, net }
    }

    /// Forward pass: returns the state value estimate.
    pub fn forward(&self, state: &Tensor) -> Tensor {
        self.net.forward(state).squeeze_dim(-1)
    }

    /// Returns a mutable reference to the variable store for optimization.
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Returns a reference to the variable store.
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
}

/// Neural network policy wrapping an [`ActorNetwork`].
///
/// Implements the [`Policy`] trait for use in the RL environment.
/// Can operate in greedy (argmax) or stochastic (sample) mode.
pub struct NeuralPolicy {
    actor: ActorNetwork,
    greedy: bool,
}

impl NeuralPolicy {
    /// Creates a new neural policy.
    ///
    /// # Arguments
    ///
    /// * `config` - Environment config (determines observation/action dimensions)
    /// * `device` - Device to run on (CPU or CUDA)
    /// * `greedy` - If true, uses argmax; if false, samples from distribution
    pub fn new(config: &RLConfig, device: Device) -> Self {
        let obs_dim = config.observation_dim();
        let action_dim = config.action_dim();
        Self {
            actor: ActorNetwork::new(obs_dim, action_dim, device),
            greedy: false,
        }
    }

    /// Sets greedy mode (argmax vs sampling).
    pub fn set_greedy(&mut self, greedy: bool) {
        self.greedy = greedy;
    }

    /// Returns a reference to the underlying actor network.
    pub fn actor(&self) -> &ActorNetwork {
        &self.actor
    }

    /// Returns a mutable reference to the underlying actor network.
    pub fn actor_mut(&mut self) -> &mut ActorNetwork {
        &mut self.actor
    }
}

impl Policy for NeuralPolicy {
    fn select_actions(&mut self, observations: &[Vec<f64>]) -> Vec<usize> {
        let n_agents = observations.len();
        if n_agents == 0 {
            return vec![];
        }

        let obs_dim = observations[0].len();
        let flat: Vec<f64> = observations
            .iter()
            .flat_map(|o| o.iter().copied())
            .collect();
        let obs_tensor = Tensor::from_slice(&flat).reshape([n_agents as i64, obs_dim as i64]);

        let actions = if self.greedy {
            let log_probs = self.actor.forward(&obs_tensor);
            log_probs.argmax(-1, false)
        } else {
            let (actions, _) = self.actor.sample_actions(&obs_tensor);
            actions
        };

        let actions_vec: Vec<i64> = actions.into();
        actions_vec.iter().map(|&a| a as usize).collect()
    }

    fn name(&self) -> &str {
        "neural"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_forward_shape() {
        let actor = ActorNetwork::new(55, 6, Device::Cpu);
        let obs = Tensor::randn([4, 55], (Kind::Float, Device::Cpu));
        let log_probs = actor.forward(&obs);
        assert_eq!(log_probs.size(), &[4, 6]);
    }

    #[test]
    fn critic_forward_shape() {
        let critic = CriticNetwork::new(100, Device::Cpu);
        let state = Tensor::randn([4, 100], (Kind::Float, Device::Cpu));
        let values = critic.forward(&state);
        assert_eq!(values.size(), &[4]);
    }

    #[test]
    fn neural_policy_select_actions() {
        let config = RLConfig::default();
        let mut policy = NeuralPolicy::new(&config, Device::Cpu);
        let obs = vec![vec![0.5; config.observation_dim()]; 3];
        let actions = policy.select_actions(&obs);
        assert_eq!(actions.len(), 3);
        for a in &actions {
            assert!(*a < config.action_dim());
        }
    }
}
