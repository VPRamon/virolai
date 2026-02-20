//! Rollout buffer for storing episode transitions.

/// A single transition stored in the buffer.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Per-agent observations.
    pub observations: Vec<Vec<f64>>,
    /// Per-agent actions.
    pub actions: Vec<usize>,
    /// Cooperative reward.
    pub reward: f64,
    /// Per-agent log-probabilities of chosen actions.
    pub log_probs: Vec<f64>,
    /// Value estimate from centralized critic.
    pub value: f64,
    /// Whether this was the last step of the episode.
    pub done: bool,
    /// Global state (for centralized critic).
    pub global_state: Vec<f64>,
}

/// Rollout buffer that stores transitions for PPO updates.
///
/// Accumulates transitions across multiple episodes, then provides
/// them for advantage computation and policy updates.
#[derive(Debug)]
pub struct RolloutBuffer {
    /// Stored transitions.
    pub transitions: Vec<Transition>,
    /// Computed advantages (populated by GAE).
    pub advantages: Vec<f64>,
    /// Computed returns (populated by GAE).
    pub returns: Vec<f64>,
}

impl RolloutBuffer {
    /// Creates a new empty buffer.
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    /// Adds a transition to the buffer.
    pub fn add(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Clears all stored data.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.advantages.clear();
        self.returns.clear();
    }

    /// Returns the number of stored transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Sets advantages and returns (called after GAE computation).
    pub fn set_advantages_and_returns(&mut self, advantages: Vec<f64>, returns: Vec<f64>) {
        assert_eq!(advantages.len(), self.transitions.len());
        assert_eq!(returns.len(), self.transitions.len());
        self.advantages = advantages;
        self.returns = returns;
    }

    /// Normalizes advantages to zero mean and unit variance.
    pub fn normalize_advantages(&mut self) {
        if self.advantages.is_empty() {
            return;
        }
        let mean: f64 = self.advantages.iter().sum::<f64>() / self.advantages.len() as f64;
        let var: f64 = self
            .advantages
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f64>()
            / self.advantages.len() as f64;
        let std = (var + 1e-8).sqrt();
        for a in &mut self.advantages {
            *a = (*a - mean) / std;
        }
    }
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_add_and_clear() {
        let mut buf = RolloutBuffer::new();
        assert!(buf.is_empty());

        buf.add(Transition {
            observations: vec![vec![0.0; 5]],
            actions: vec![0],
            reward: 1.0,
            log_probs: vec![-0.5],
            value: 0.5,
            done: false,
            global_state: vec![0.0; 10],
        });

        assert_eq!(buf.len(), 1);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn normalize_advantages() {
        let mut buf = RolloutBuffer::new();
        for _ in 0..3 {
            buf.add(Transition {
                observations: vec![],
                actions: vec![],
                reward: 0.0,
                log_probs: vec![],
                value: 0.0,
                done: false,
                global_state: vec![],
            });
        }
        buf.set_advantages_and_returns(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        buf.normalize_advantages();

        let mean: f64 = buf.advantages.iter().sum::<f64>() / buf.advantages.len() as f64;
        assert!(mean.abs() < 1e-6);
    }
}
