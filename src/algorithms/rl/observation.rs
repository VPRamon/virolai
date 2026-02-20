//! Observation encoding for the RL environment.
//!
//! Builds per-agent observation vectors containing the agent's own state
//! plus features for the Top-M candidate tasks.

use super::agent::AgentState;
use super::config::RLConfig;
use super::task_pool::{TaskInstance, TaskPool};

/// Builds observation vectors for agents.
pub struct ObservationBuilder;

impl ObservationBuilder {
    /// Builds the observation vector for a specific agent.
    ///
    /// The observation is a flat `Vec<f64>` with structure:
    /// ```text
    /// [agent_features(5)] ++ [task_1_features(10)] ++ ... ++ [task_M_features(10)]
    /// ```
    ///
    /// If fewer than M tasks are active, remaining slots are zero-padded.
    ///
    /// # Arguments
    ///
    /// * `agent_idx` - Index of the agent to build observation for
    /// * `agents` - All agents (needed for heading counts and distance computation)
    /// * `task_pool` - Active task pool
    /// * `config` - Environment configuration
    pub fn build(
        agent_idx: usize,
        agents: &[AgentState],
        task_pool: &TaskPool,
        config: &RLConfig,
    ) -> Vec<f64> {
        let agent = &agents[agent_idx];
        let agent_positions: Vec<_> = agents.iter().map(|a| a.position).collect();
        let top_m = task_pool.top_m(&agent_positions, config.top_m);
        let max_value = task_pool
            .active
            .iter()
            .map(|t| t.value)
            .fold(1.0_f64, f64::max);

        let mut obs = agent.features(config);

        for i in 0..config.top_m {
            if i < top_m.len() {
                let task = top_m[i];
                let heading_counts = Self::count_heading_agents(agents, task);
                obs.extend(task.features(config, max_value, heading_counts));
            } else {
                // Zero-padding for missing tasks
                obs.extend(std::iter::repeat_n(0.0, RLConfig::TASK_FEATURE_DIM));
            }
        }

        obs
    }

    /// Builds observations for all agents.
    pub fn build_all(
        agents: &[AgentState],
        task_pool: &TaskPool,
        config: &RLConfig,
    ) -> Vec<Vec<f64>> {
        (0..agents.len())
            .map(|i| Self::build(i, agents, task_pool, config))
            .collect()
    }

    /// Counts how many agents of each type are currently heading toward a task.
    ///
    /// "Heading toward" means the agent's current_target matches the task's
    /// index in the Top-M list. This is an approximation since agents may
    /// have different Top-M orderings, but it provides useful signal for
    /// coordination.
    fn count_heading_agents(agents: &[AgentState], task: &TaskInstance) -> [u32; 3] {
        let mut counts = [0u32; 3];
        for agent in agents {
            // An agent is heading toward this task if they are within
            // the task's collection radius (a proxy for "assigned to this task")
            if agent.position.distance_to(&task.position) <= task.collection_radius * 3.0 {
                counts[agent.agent_type.index()] += 1;
            }
        }
        counts
    }

    /// Returns the global state vector (used by the centralized critic).
    ///
    /// Concatenates all agent features and all active task features.
    pub fn build_global_state(
        agents: &[AgentState],
        task_pool: &TaskPool,
        config: &RLConfig,
    ) -> Vec<f64> {
        let max_value = task_pool
            .active
            .iter()
            .map(|t| t.value)
            .fold(1.0_f64, f64::max);

        let mut state = Vec::new();

        // All agent features
        for agent in agents {
            state.extend(agent.features(config));
        }

        // All active task features (up to max_active_tasks, zero-padded)
        let agent_positions: Vec<_> = agents.iter().map(|a| a.position).collect();
        let top_m = task_pool.top_m(&agent_positions, config.max_active_tasks);
        for i in 0..config.max_active_tasks {
            if i < top_m.len() {
                let heading = Self::count_heading_agents(agents, top_m[i]);
                state.extend(top_m[i].features(config, max_value, heading));
            } else {
                state.extend(std::iter::repeat_n(0.0, RLConfig::TASK_FEATURE_DIM));
            }
        }

        state
    }

    /// Dimension of the global state vector.
    pub fn global_state_dim(n_agents: usize, config: &RLConfig) -> usize {
        n_agents * RLConfig::AGENT_FEATURE_DIM
            + config.max_active_tasks * RLConfig::TASK_FEATURE_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rl::types::{AgentType, AgentTypeRequirements, Position};

    fn make_agents() -> Vec<AgentState> {
        vec![
            AgentState::new("a0".into(), Position::new(1.0, 1.0), AgentType::Young),
            AgentState::new("a1".into(), Position::new(5.0, 5.0), AgentType::Old),
        ]
    }

    fn make_pool_with_tasks() -> TaskPool {
        let mut pool = TaskPool::new(vec![]);
        pool.active.push(TaskInstance {
            id: "t0".into(),
            position: Position::new(2.0, 2.0),
            value: 5.0,
            deadline: 10,
            remaining_time: 8,
            type_requirements: AgentTypeRequirements::new(1, 0, 0),
            collection_radius: 1.0,
            remaining_appearances: None,
        });
        pool
    }

    #[test]
    fn observation_has_correct_dim() {
        let agents = make_agents();
        let pool = make_pool_with_tasks();
        let config = RLConfig::default();
        let obs = ObservationBuilder::build(0, &agents, &pool, &config);
        assert_eq!(obs.len(), config.observation_dim());
    }

    #[test]
    fn all_observations_same_dim() {
        let agents = make_agents();
        let pool = make_pool_with_tasks();
        let config = RLConfig::default();
        let all_obs = ObservationBuilder::build_all(&agents, &pool, &config);
        assert_eq!(all_obs.len(), agents.len());
        for obs in &all_obs {
            assert_eq!(obs.len(), config.observation_dim());
        }
    }

    #[test]
    fn global_state_dim_correct() {
        let agents = make_agents();
        let pool = make_pool_with_tasks();
        let config = RLConfig::default();
        let state = ObservationBuilder::build_global_state(&agents, &pool, &config);
        assert_eq!(
            state.len(),
            ObservationBuilder::global_state_dim(agents.len(), &config)
        );
    }
}
