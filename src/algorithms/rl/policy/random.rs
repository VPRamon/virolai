//! Random policy for testing and baselines.

use rand::Rng;

use super::trait_::Policy;

/// Uniformly random action selection.
///
/// Each agent independently selects a random action from `[0, action_dim)`.
/// Used for sanity checks and as a lower-bound baseline.
pub struct RandomPolicy {
    action_dim: usize,
}

impl RandomPolicy {
    /// Creates a new random policy.
    ///
    /// # Arguments
    ///
    /// * `action_dim` - Number of possible actions (Top-M + 1 for patrol).
    pub fn new(action_dim: usize) -> Self {
        Self { action_dim }
    }
}

impl Policy for RandomPolicy {
    fn select_actions(&mut self, observations: &[Vec<f64>]) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        (0..observations.len())
            .map(|_| rng.gen_range(0..self.action_dim))
            .collect()
    }

    fn name(&self) -> &str {
        "random"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_policy_returns_correct_count() {
        let mut policy = RandomPolicy::new(6);
        let obs = vec![vec![0.0; 10]; 4]; // 4 agents
        let actions = policy.select_actions(&obs);
        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn random_policy_actions_in_range() {
        let mut policy = RandomPolicy::new(6);
        let obs = vec![vec![0.0; 10]; 100];
        let actions = policy.select_actions(&obs);
        for a in actions {
            assert!(a < 6);
        }
    }
}
