//! Policy trait for the RL environment.

/// A policy that selects actions for agents based on observations.
///
/// Actions are indices into the Top-M + patrol set:
/// - 0: patrol (no target)
/// - 1..=M: target the corresponding task in the Top-M list
pub trait Policy: Send + Sync {
    /// Selects one action per agent given their observations.
    ///
    /// # Arguments
    ///
    /// * `observations` - Per-agent observation vectors (from [`ObservationBuilder`])
    ///
    /// # Returns
    ///
    /// A vector of actions, one per agent.
    fn select_actions(&mut self, observations: &[Vec<f64>]) -> Vec<usize>;

    /// Returns a human-readable name for this policy.
    fn name(&self) -> &str;
}
