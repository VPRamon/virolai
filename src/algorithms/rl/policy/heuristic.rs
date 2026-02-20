//! Greedy heuristic policy (§7.1 baseline).
//!
//! Assigns agents to tasks by scoring each task based on value, distance,
//! and urgency. Fulfills type requirements greedily, allowing excess agents.

use super::trait_::Policy;
use crate::algorithms::rl::config::RLConfig;

/// Greedy heuristic policy for multi-agent task assignment.
///
/// For each agent, scores available tasks by:
/// ```text
/// score(j) = V_j / (ε + distance(agent, task_j)) × 1 / (ε + d_j)
/// ```
///
/// Assigns agents to tasks greedily, prioritizing tasks that are urgent
/// and valuable. Tracks assignment counts per task to satisfy type
/// requirements before allowing over-assignment.
///
/// This policy serves as a competitive baseline (§7.1) and should outperform
/// the random policy significantly.
pub struct GreedyHeuristicPolicy {
    config: RLConfig,
}

impl GreedyHeuristicPolicy {
    /// Creates a new greedy heuristic policy with the given configuration.
    pub fn new(config: RLConfig) -> Self {
        Self { config }
    }
}

impl Policy for GreedyHeuristicPolicy {
    fn select_actions(&mut self, observations: &[Vec<f64>]) -> Vec<usize> {
        let n_agents = observations.len();
        let top_m = self.config.top_m;
        let eps = 1e-6;

        // Track how many agents are heading to each task (for coordination)
        let mut assigned_counts = vec![0u32; top_m];
        let mut actions = vec![0usize; n_agents];

        // First pass: extract per-agent and per-task info from observations
        // Observation layout: [agent_features(5)] ++ [task_1(10)] ++ ... ++ [task_M(10)]
        let agent_feat_dim = RLConfig::AGENT_FEATURE_DIM;
        let task_feat_dim = RLConfig::TASK_FEATURE_DIM;

        for i in 0..n_agents {
            let obs = &observations[i];
            if obs.len() < agent_feat_dim {
                actions[i] = 0;
                continue;
            }

            let agent_x = obs[0]; // normalized
            let agent_y = obs[1]; // normalized

            // Determine agent type from one-hot encoding
            let agent_type_idx = if obs[2] > 0.5 {
                0 // young
            } else if obs[3] > 0.5 {
                1 // middle
            } else {
                2 // old
            };

            let mut best_score = f64::NEG_INFINITY;
            let mut best_task = 0usize; // 0 = patrol

            for (j, task_obs) in obs[agent_feat_dim..]
                .chunks(task_feat_dim)
                .take(top_m)
                .enumerate()
            {
                if task_obs.len() < task_feat_dim {
                    break;
                }

                let task_x = task_obs[0]; // normalized
                let task_y = task_obs[1];
                let task_value = task_obs[2]; // normalized
                let task_time_left = task_obs[3]; // normalized
                let task_reqs = [
                    task_obs[4], // r_young
                    task_obs[5], // r_middle
                    task_obs[6], // r_old
                ];

                // Skip zero-padded (empty) task slots
                if task_value.abs() < eps && task_time_left.abs() < eps {
                    continue;
                }

                // Distance in normalized space
                let dx = agent_x - task_x;
                let dy = agent_y - task_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Urgency factor: prefer tasks about to expire
                let urgency = 1.0 / (eps + task_time_left);

                // Base score
                let mut score = (task_value / (eps + dist)) * urgency;

                // Bonus for tasks that specifically need this agent's type
                let my_req = task_reqs[agent_type_idx];
                if my_req > 0.0 {
                    // This task needs agents of my type
                    let already_assigned_my_type = assigned_counts[j]; // rough approximation
                    if (already_assigned_my_type as f64) < my_req {
                        score *= 2.0; // boost for unfulfilled requirements
                    }
                }

                if score > best_score {
                    best_score = score;
                    best_task = j + 1; // 1-indexed
                }
            }

            actions[i] = best_task;
            if best_task > 0 {
                assigned_counts[best_task - 1] += 1;
            }
        }

        actions
    }

    fn name(&self) -> &str {
        "greedy_heuristic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_returns_correct_count() {
        let config = RLConfig::default();
        let mut policy = GreedyHeuristicPolicy::new(config.clone());
        // Create observations with proper dimensions
        let obs_dim = config.observation_dim();
        let obs = vec![vec![0.5; obs_dim]; 4]; // 4 agents
        let actions = policy.select_actions(&obs);
        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn heuristic_selects_urgent_task() {
        let config = RLConfig {
            top_m: 2,
            ..RLConfig::default()
        };
        let mut policy = GreedyHeuristicPolicy::new(config);

        // Agent at (0.5, 0.5)
        let mut obs = vec![0.0; 5 + 2 * 10];
        obs[0] = 0.5; // x
        obs[1] = 0.5; // y
        obs[2] = 1.0; // young

        // Task 1: low urgency, high value, close
        obs[5] = 0.5; // x
        obs[6] = 0.5; // y
        obs[7] = 0.9; // value
        obs[8] = 0.9; // time_left (lots of time = low urgency)
        obs[9] = 1.0; // r_young

        // Task 2: high urgency, moderate value, close
        obs[15] = 0.5; // x
        obs[16] = 0.5; // y
        obs[17] = 0.5; // value
        obs[18] = 0.01; // time_left (very urgent!)
        obs[19] = 1.0; // r_young

        let actions = policy.select_actions(&[obs]);
        assert_eq!(actions[0], 2); // should pick the urgent task
    }
}
