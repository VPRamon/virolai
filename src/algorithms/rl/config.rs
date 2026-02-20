//! Configuration for the RL scheduling environment and training.

use std::collections::HashMap;

use super::types::AgentType;

/// Configuration for the RL scheduling environment.
///
/// Controls environment geometry, agent dynamics, task spawning,
/// observation encoding, and reward shaping.
#[derive(Debug, Clone)]
pub struct RLConfig {
    // --- Environment geometry ---
    /// Width of the 2D domain Ω.
    pub world_width: f64,
    /// Height of the 2D domain Ω.
    pub world_height: f64,
    /// Episode horizon (number of time steps T).
    pub episode_horizon: u32,
    /// Duration of one time step Δt.
    pub delta_t: f64,

    // --- Agent dynamics ---
    /// Maximum speed per agent type (displacement per Δt).
    pub speeds: HashMap<AgentType, f64>,

    // --- Task parameters ---
    /// Global default collection radius ρ.
    pub collection_radius: f64,
    /// Task spawn probability per time step λ.
    pub spawn_rate: f64,
    /// Maximum number of simultaneously active tasks.
    pub max_active_tasks: usize,

    // --- Observation ---
    /// Number of Top-M candidate tasks to include in observations.
    pub top_m: usize,

    // --- Reward shaping ---
    /// Time penalty per step c_time (subtracted each step).
    pub reward_time_penalty: f64,
    /// Progress shaping coefficient α.
    pub reward_progress_alpha: f64,
    /// Expiration penalty coefficient β (penalty = β × V_j).
    pub reward_expiry_beta: f64,
    /// Coverage shaping coefficient γ.
    pub reward_coverage_gamma: f64,

    // --- Discount ---
    /// Discount factor for RL returns.
    pub gamma: f64,
}

impl RLConfig {
    /// Returns the maximum speed for a given agent type.
    pub fn speed_for(&self, agent_type: AgentType) -> f64 {
        self.speeds
            .get(&agent_type)
            .copied()
            .unwrap_or_else(|| agent_type.default_max_speed())
    }

    /// Observation dimension per agent: own features + top_m × task features.
    pub fn observation_dim(&self) -> usize {
        Self::AGENT_FEATURE_DIM + self.top_m * Self::TASK_FEATURE_DIM
    }

    /// Number of features encoding a single agent.
    pub const AGENT_FEATURE_DIM: usize = 5; // x, y, one_hot(3)

    /// Number of features encoding a single task candidate.
    pub const TASK_FEATURE_DIM: usize = 10; // x, y, value, time_left, r_young, r_middle, r_old, k_rem, heading_young, heading_middle (+ heading_old implied)

    /// Number of possible actions: 0 = patrol, 1..=top_m = target task.
    pub fn action_dim(&self) -> usize {
        self.top_m + 1
    }
}

impl Default for RLConfig {
    fn default() -> Self {
        let mut speeds = HashMap::new();
        speeds.insert(AgentType::Young, 3.0);
        speeds.insert(AgentType::Middle, 2.0);
        speeds.insert(AgentType::Old, 1.0);

        Self {
            world_width: 10.0,
            world_height: 10.0,
            episode_horizon: 100,
            delta_t: 1.0,
            speeds,
            collection_radius: 1.0,
            spawn_rate: 0.3,
            max_active_tasks: 20,
            top_m: 5,
            reward_time_penalty: 0.01,
            reward_progress_alpha: 0.1,
            reward_expiry_beta: 0.5,
            reward_coverage_gamma: 0.05,
            gamma: 0.99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = RLConfig::default();
        assert!(cfg.world_width > 0.0);
        assert!(cfg.world_height > 0.0);
        assert!(cfg.episode_horizon > 0);
        assert!(cfg.top_m > 0);
        assert_eq!(cfg.action_dim(), cfg.top_m + 1);
    }

    #[test]
    fn observation_dim_matches() {
        let cfg = RLConfig::default();
        let expected = 5 + cfg.top_m * 10;
        assert_eq!(cfg.observation_dim(), expected);
    }

    #[test]
    fn speed_for_known_type() {
        let cfg = RLConfig::default();
        assert_eq!(cfg.speed_for(AgentType::Young), 3.0);
        assert_eq!(cfg.speed_for(AgentType::Old), 1.0);
    }
}
