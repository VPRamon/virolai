//! Composite reward function for the RL environment.
//!
//! Combines collection rewards, time penalties, progress shaping,
//! requirement coverage shaping, and expiration penalties.

use super::agent::AgentState;
use super::config::RLConfig;
use super::task_pool::TaskInstance;

/// Computes rewards for the RL environment.
pub struct RewardComputer;

impl RewardComputer {
    /// Computes the total reward for a single environment step.
    ///
    /// # Components
    ///
    /// 1. **Collection reward**: `+V_j` for each collected task.
    /// 2. **Expiration penalty**: `-β × V_j` for each expired task.
    /// 3. **Time penalty**: `-c_time` per step.
    /// 4. **Progress shaping**: `+α × Σ_i (dist_prev - dist_now)` for agents moving toward targets.
    /// 5. **Coverage shaping**: `+γ × ΔP_j` when requirement coverage improves.
    pub fn compute(
        collected_value: f64,
        expired_tasks: &[TaskInstance],
        agents: &[AgentState],
        top_m_tasks: &[&TaskInstance],
        config: &RLConfig,
    ) -> f64 {
        let mut reward = 0.0;

        // 1. Collection reward
        reward += collected_value;

        // 2. Expiration penalty
        for task in expired_tasks {
            reward -= config.reward_expiry_beta * task.value;
        }

        // 3. Time penalty
        reward -= config.reward_time_penalty;

        // 4. Progress shaping
        let progress: f64 = agents
            .iter()
            .map(|a| a.progress_toward_target(top_m_tasks))
            .sum();
        reward += config.reward_progress_alpha * progress;

        reward
    }

    /// Computes requirement coverage progress for a task.
    ///
    /// P_j = Σ_type (r_type × min(1, c_type / r_type)) / Σ_type r_type
    ///
    /// where c_type is the count of agents of that type within collection radius.
    pub fn coverage_progress(
        task: &TaskInstance,
        agent_positions: &[(super::types::Position, super::types::AgentType)],
    ) -> f64 {
        let reqs = task.type_requirements.as_array();
        let total_req: u32 = reqs.iter().sum();
        if total_req == 0 {
            return 1.0;
        }

        let mut counts = [0u32; 3];
        for (pos, atype) in agent_positions {
            if pos.distance_to(&task.position) <= task.collection_radius {
                counts[atype.index()] += 1;
            }
        }

        let mut progress = 0.0;
        for i in 0..3 {
            if reqs[i] > 0 {
                let ratio = (counts[i] as f64 / reqs[i] as f64).min(1.0);
                progress += reqs[i] as f64 * ratio;
            }
        }

        progress / total_req as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rl::types::{AgentType, AgentTypeRequirements, Position};

    #[test]
    fn collection_reward_added() {
        let config = RLConfig::default();
        let reward = RewardComputer::compute(10.0, &[], &[], &[], &config);
        // Should be: 10.0 - c_time
        assert!(reward > 9.0);
    }

    #[test]
    fn expiry_penalty_subtracted() {
        let config = RLConfig::default();
        let expired = vec![TaskInstance {
            id: "t".into(),
            position: Position::new(0.0, 0.0),
            value: 10.0,
            deadline: 1,
            remaining_time: 0,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        }];
        let reward = RewardComputer::compute(0.0, &expired, &[], &[], &config);
        // Should be: -beta*10 - c_time = -5.0 - 0.01
        assert!(reward < -4.0);
    }

    #[test]
    fn coverage_progress_full() {
        let task = TaskInstance {
            id: "t".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 5,
            type_requirements: AgentTypeRequirements::new(1, 1, 0),
            collection_radius: 2.0,
            remaining_appearances: None,
        };
        let agents = vec![
            (Position::new(5.0, 5.0), AgentType::Young),
            (Position::new(5.0, 5.0), AgentType::Middle),
        ];
        let p = RewardComputer::coverage_progress(&task, &agents);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn coverage_progress_partial() {
        let task = TaskInstance {
            id: "t".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 5,
            type_requirements: AgentTypeRequirements::new(2, 0, 0),
            collection_radius: 2.0,
            remaining_appearances: None,
        };
        // Only 1 of 2 required young agents present
        let agents = vec![(Position::new(5.0, 5.0), AgentType::Young)];
        let p = RewardComputer::coverage_progress(&task, &agents);
        assert!((p - 0.5).abs() < 1e-10);
    }
}
