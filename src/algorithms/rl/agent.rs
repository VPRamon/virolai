//! Agent state and dynamics for the RL environment.

use super::config::RLConfig;
use super::task_pool::TaskInstance;
use super::types::{AgentType, Position};
use crate::Id;

/// State of a single agent in the RL environment.
///
/// Each agent has a position in the 2D domain, a type (which determines its
/// maximum movement speed), and an optional current target task.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Unique identifier for this agent.
    pub id: Id,
    /// Current position in the 2D domain.
    pub position: Position,
    /// Agent type (determines max speed).
    pub agent_type: AgentType,
    /// Index into the Top-M candidate list (0 = patrol / no target).
    pub current_target: usize,
    /// Distance to target at end of previous step (for progress shaping).
    pub prev_distance_to_target: Option<f64>,
}

impl AgentState {
    /// Creates a new agent.
    pub fn new(id: Id, position: Position, agent_type: AgentType) -> Self {
        Self {
            id,
            position,
            agent_type,
            current_target: 0,
            prev_distance_to_target: None,
        }
    }

    /// Applies an action: sets the current target and moves toward it.
    ///
    /// - Action 0: patrol (no target; agent stays in place or wanders).
    /// - Action 1..=M: index into the Top-M candidate task list.
    ///
    /// The agent moves toward the target task's position at most
    /// `v(τ) × Δt` distance, clamped to domain bounds.
    pub fn step(&mut self, action: usize, top_m_tasks: &[&TaskInstance], config: &RLConfig) {
        self.current_target = action;
        let max_dist = config.speed_for(self.agent_type) * config.delta_t;

        if action == 0 || action > top_m_tasks.len() {
            // Patrol: no movement (agent holds position)
            self.prev_distance_to_target = None;
            return;
        }

        let task = &top_m_tasks[action - 1];
        let target = task.position;
        self.prev_distance_to_target = Some(self.position.distance_to(&target));
        self.position
            .move_toward(&target, max_dist, config.world_width, config.world_height);
    }

    /// Returns the distance to the current target task, if any.
    pub fn distance_to_target(&self, top_m_tasks: &[&TaskInstance]) -> Option<f64> {
        if self.current_target == 0 || self.current_target > top_m_tasks.len() {
            return None;
        }
        let task = &top_m_tasks[self.current_target - 1];
        Some(self.position.distance_to(&task.position))
    }

    /// Computes progress toward current target (positive = got closer).
    pub fn progress_toward_target(&self, top_m_tasks: &[&TaskInstance]) -> f64 {
        match (
            self.prev_distance_to_target,
            self.distance_to_target(top_m_tasks),
        ) {
            (Some(prev), Some(now)) => prev - now,
            _ => 0.0,
        }
    }

    /// Encodes agent state as a feature vector for observation.
    ///
    /// Returns `[x_norm, y_norm, one_hot_young, one_hot_middle, one_hot_old]`.
    pub fn features(&self, config: &RLConfig) -> Vec<f64> {
        let (nx, ny) = self
            .position
            .normalized(config.world_width, config.world_height);
        let oh = self.agent_type.one_hot();
        vec![nx, ny, oh[0], oh[1], oh[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RLConfig {
        RLConfig::default()
    }

    #[test]
    fn agent_features_length() {
        let agent = AgentState::new("a1".into(), Position::new(5.0, 5.0), AgentType::Young);
        let config = default_config();
        let features = agent.features(&config);
        assert_eq!(features.len(), RLConfig::AGENT_FEATURE_DIM);
    }

    #[test]
    fn agent_features_normalized() {
        let config = default_config();
        let agent = AgentState::new(
            "a1".into(),
            Position::new(config.world_width, config.world_height),
            AgentType::Old,
        );
        let features = agent.features(&config);
        assert!((features[0] - 1.0).abs() < 1e-10);
        assert!((features[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn step_patrol_no_movement() {
        let config = default_config();
        let mut agent = AgentState::new("a1".into(), Position::new(5.0, 5.0), AgentType::Young);
        let original = agent.position;
        agent.step(0, &[], &config);
        assert_eq!(agent.position.x, original.x);
        assert_eq!(agent.position.y, original.y);
    }

    #[test]
    fn step_toward_task() {
        use super::super::types::AgentTypeRequirements;
        let config = default_config();
        let mut agent = AgentState::new("a1".into(), Position::new(0.0, 0.0), AgentType::Young);
        let task = TaskInstance {
            id: "t1".into(),
            position: Position::new(10.0, 0.0),
            value: 1.0,
            deadline: 10,
            remaining_time: 10,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        };
        agent.step(1, &[&task], &config);
        // Young agent speed = 3.0, should move 3 units toward (10, 0)
        assert!((agent.position.x - 3.0).abs() < 1e-10);
    }
}
