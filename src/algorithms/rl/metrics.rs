//! Evaluation metrics for the RL scheduling environment (§10).
//!
//! Tracks episode-level performance metrics and provides aggregation
//! over multiple evaluation episodes.

use std::fmt;

use super::environment::RLEnvironment;
use super::policy::Policy;

/// Aggregated evaluation metrics over multiple episodes.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Mean total collected value per episode.
    pub mean_collected_value: f64,
    /// Mean percentage of tasks collected before deadline.
    pub mean_pct_collected: f64,
    /// Mean value lost to expirations per episode.
    pub mean_value_lost: f64,
    /// Mean cumulative reward per episode.
    pub mean_cumulative_reward: f64,
    /// Mean number of tasks collected per episode.
    pub mean_tasks_collected: f64,
    /// Mean number of tasks expired per episode.
    pub mean_tasks_expired: f64,
    /// Number of episodes evaluated.
    pub n_episodes: usize,
}

/// Tracks per-episode statistics during evaluation.
#[derive(Debug, Default)]
struct EpisodeStats {
    total_collected_value: f64,
    total_expired_value: f64,
    tasks_collected: u32,
    tasks_expired: u32,
    cumulative_reward: f64,
}

impl EvaluationMetrics {
    /// Evaluates a policy over multiple episodes and returns aggregated metrics.
    ///
    /// # Arguments
    ///
    /// * `env` - The RL environment to evaluate in
    /// * `policy` - The policy to evaluate
    /// * `n_episodes` - Number of episodes to run
    pub fn evaluate(env: &mut RLEnvironment, policy: &mut dyn Policy, n_episodes: usize) -> Self {
        let mut all_stats = Vec::with_capacity(n_episodes);

        for _ in 0..n_episodes {
            let mut obs = env.reset();
            let mut stats = EpisodeStats::default();

            loop {
                let actions = policy.select_actions(&obs);
                let result = env.step(actions);

                stats.tasks_collected += result.tasks_collected as u32;
                stats.tasks_expired += result.tasks_expired as u32;

                // Track collected value (embedded in reward before shaping)
                // We can approximate it from reward + penalties
                if result.tasks_collected > 0 {
                    // The collection value is the positive component
                    // We track it through the task pool's state
                    stats.total_collected_value += result.reward.max(0.0);
                }

                obs = result.observations;

                if result.done {
                    stats.cumulative_reward = env.cumulative_reward;
                    break;
                }
            }

            all_stats.push(stats);
        }

        let n = all_stats.len() as f64;
        let mean_collected_value = all_stats
            .iter()
            .map(|s| s.total_collected_value)
            .sum::<f64>()
            / n;
        let mean_value_lost = all_stats.iter().map(|s| s.total_expired_value).sum::<f64>() / n;
        let mean_tasks_collected = all_stats
            .iter()
            .map(|s| s.tasks_collected as f64)
            .sum::<f64>()
            / n;
        let mean_tasks_expired = all_stats
            .iter()
            .map(|s| s.tasks_expired as f64)
            .sum::<f64>()
            / n;
        let mean_cumulative_reward = all_stats.iter().map(|s| s.cumulative_reward).sum::<f64>() / n;

        let mean_pct_collected = all_stats
            .iter()
            .map(|s| {
                let total = s.tasks_collected + s.tasks_expired;
                if total > 0 {
                    s.tasks_collected as f64 / total as f64 * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / n;

        Self {
            mean_collected_value,
            mean_pct_collected,
            mean_value_lost,
            mean_cumulative_reward,
            mean_tasks_collected,
            mean_tasks_expired,
            n_episodes,
        }
    }
}

impl fmt::Display for EvaluationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== Evaluation Metrics ({} episodes) ===",
            self.n_episodes
        )?;
        writeln!(
            f,
            "  Mean collected value:    {:.2}",
            self.mean_collected_value
        )?;
        writeln!(
            f,
            "  Mean tasks collected:    {:.1}",
            self.mean_tasks_collected
        )?;
        writeln!(
            f,
            "  Mean tasks expired:      {:.1}",
            self.mean_tasks_expired
        )?;
        writeln!(
            f,
            "  Mean % collected:        {:.1}%",
            self.mean_pct_collected
        )?;
        writeln!(f, "  Mean value lost:         {:.2}", self.mean_value_lost)?;
        writeln!(
            f,
            "  Mean cumulative reward:  {:.2}",
            self.mean_cumulative_reward
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::rl::{AgentType, RLConfig, RandomPolicy};

    #[test]
    fn evaluate_completes() {
        let config = RLConfig {
            episode_horizon: 10,
            ..RLConfig::default()
        };
        let mut env = RLEnvironment::new(config.clone(), 42);
        env.set_agents(&[(2, AgentType::Young), (1, AgentType::Old)]);
        let mut policy = RandomPolicy::new(config.action_dim());
        let metrics = EvaluationMetrics::evaluate(&mut env, &mut policy, 3);
        assert_eq!(metrics.n_episodes, 3);
    }
}
