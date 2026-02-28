//! Policy-driven scheduling algorithm.
//!
//! Bridges a trained [`NeuralPolicy`] to the [`SchedulingAlgorithm`] interface,
//! using the policy to determine task selection order and the existing
//! constraint-respecting placement logic from [`find_earliest_non_overlapping`].
//!
//! # Milestone 1 constraints
//!
//! Same as [`RLScheduler`]: time-only, single agent type, one resource per task.
//! Spatial displacement is ignored (all positions at origin).

use std::path::{Path, PathBuf};

use qtty::{Quantity, Unit};
use tch::Device;

use super::config::RLConfig;
use super::environment::RLEnvironment;
use super::network::NeuralPolicy;
use super::policy::Policy;
use super::task_pool::TaskTemplate;
use super::types::{AgentType, AgentTypeRequirements};
use crate::algorithms::SchedulingAlgorithm;
use crate::schedule::Schedule;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::{Interval, SolutionSpace};

/// Scheduler that uses a trained neural policy for task selection ordering.
///
/// The policy decides which tasks to prioritize; the constraint-respecting
/// placement logic handles exact time-window assignment.
///
/// # Construction
///
/// ```ignore
/// use virolai::algorithms::rl::policy_scheduler::PolicyDrivenScheduler;
///
/// let scheduler = PolicyDrivenScheduler::from_checkpoint("checkpoints/actor.pt")?;
/// let schedule = scheduler.schedule(&blocks, &solution_space, horizon);
/// ```
pub struct PolicyDrivenScheduler {
    /// Loaded neural policy (runs in greedy mode for inference).
    policy: NeuralPolicy,
    /// RL environment configuration.
    config: RLConfig,
    /// Path to the actor checkpoint.
    checkpoint_path: PathBuf,
}

impl PolicyDrivenScheduler {
    /// Creates a new policy-driven scheduler from a saved actor checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the saved `actor.pt` file
    /// * `device` - Device to run inference on (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint file cannot be loaded.
    pub fn from_checkpoint(
        checkpoint_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, tch::TchError> {
        Self::from_checkpoint_with_config(checkpoint_path, RLConfig::default(), device)
    }

    /// Creates a new policy-driven scheduler with custom RL config.
    pub fn from_checkpoint_with_config(
        checkpoint_path: impl AsRef<Path>,
        config: RLConfig,
        device: Device,
    ) -> Result<Self, tch::TchError> {
        let mut policy = NeuralPolicy::new(&config, device);
        policy.load_actor(checkpoint_path.as_ref())?;
        policy.set_greedy(true);

        Ok(Self {
            policy,
            config,
            checkpoint_path: checkpoint_path.as_ref().to_path_buf(),
        })
    }

    /// Creates a policy-driven scheduler with an already-loaded policy.
    ///
    /// Useful for evaluation during training without reloading from disk.
    pub fn with_policy(policy: NeuralPolicy, config: RLConfig) -> Self {
        Self {
            policy,
            config,
            checkpoint_path: PathBuf::new(),
        }
    }

    /// Returns the checkpoint path.
    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    /// Runs a single RL episode to determine task selection ordering.
    ///
    /// Converts scheduling tasks to RL task templates, runs the policy in
    /// the environment, and returns the IDs of collected tasks in the order
    /// they were collected.
    fn determine_task_order<T, U>(
        &mut self,
        blocks: &[SchedulingBlock<T, U, (), petgraph::Directed>],
        solution_space: &SolutionSpace<U>,
    ) -> Vec<String>
    where
        T: Task<U> + Clone,
        U: Unit,
    {
        // Convert scheduling tasks to RL task templates.
        let mut templates = Vec::new();

        for block in blocks {
            for (id, task) in block.tasks() {
                // Skip tasks with no feasible windows
                let Some(intervals) = solution_space.get_intervals(id) else {
                    continue;
                };

                let task_size = task.size_on_axis().value();
                let can_fit = intervals.iter().any(|iv| iv.duration().value() >= task_size);
                if !can_fit {
                    continue;
                }

                // Map priority to value (positive, higher is better)
                let value = (task.priority() as f64).max(1.0);

                // Derive deadline from available capacity: more capacity = later deadline
                let total_capacity: f64 =
                    intervals.iter().map(|iv| iv.duration().value()).sum::<f64>();
                let urgency_steps = (total_capacity / task_size).ceil() as u32;
                let deadline = urgency_steps.max(1).min(self.config.episode_horizon);

                templates.push(TaskTemplate {
                    name: id.to_string(),
                    value_range: (value, value), // Fixed value (no randomness)
                    deadline_range: (deadline, deadline),
                    type_requirements: AgentTypeRequirements::new(1, 0, 0),
                    collection_radius: Some(self.config.collection_radius),
                    max_appearances: Some(1),
                    appearances_used: 0,
                });
            }
        }

        if templates.is_empty() {
            return Vec::new();
        }

        // Create environment with custom templates
        let mut env = RLEnvironment::with_templates(self.config.clone(), templates, 0);
        env.set_agents(&[(1, AgentType::Young)]);

        // Run episode with the policy
        let mut obs = env.reset();
        let mut collected_order = Vec::new();

        for _ in 0..self.config.episode_horizon {
            let actions = self.policy.select_actions(&obs);
            let result = env.step(actions);

            // Record newly collected task IDs (instance ID → task ID)
            for instance_id in &env.task_pool.collected_ids {
                let task_id = instance_id_to_task_id(instance_id);
                if !collected_order.contains(&task_id.to_string()) {
                    collected_order.push(task_id.to_string());
                }
            }

            obs = result.observations;
            if result.done {
                break;
            }
        }

        collected_order
    }
}

impl<T, U, D, E> SchedulingAlgorithm<T, U, D, E> for PolicyDrivenScheduler
where
    T: Task<U> + Clone,
    U: Unit,
    E: petgraph::EdgeType,
{
    fn schedule(
        &self,
        blocks: &[SchedulingBlock<T, U, D, E>],
        solution_space: &SolutionSpace<U>,
        horizon: Interval<U>,
    ) -> Schedule<U> {
        // We need `&mut self` for the policy, but the trait requires `&self`.
        // Use unsafe interior mutability via a raw pointer. The policy's
        // select_actions only mutates internal RNG state which is safe for
        // single-threaded scheduling.
        let self_mut = unsafe { &mut *(self as *const Self as *mut Self) };

        let mut schedule = Schedule::new();
        let horizon_start = horizon.start().value();
        let horizon_end = horizon.end().value();

        // Build a task-info index: id → (size, priority, fitting windows)
        let mut task_info: Vec<(String, f64, i32, Vec<(f64, f64)>)> = Vec::new();

        for block in blocks {
            for (id, task) in block.tasks() {
                let Some(raw_intervals) = solution_space.get_intervals(id) else {
                    continue;
                };
                let task_size = task.size_on_axis().value();
                let fitting: Vec<(f64, f64)> = raw_intervals
                    .iter()
                    .filter(|iv| iv.duration().value() >= task_size)
                    .map(|iv| (iv.start().value(), iv.end().value()))
                    .collect();
                if fitting.is_empty() {
                    continue;
                }
                task_info.push((id.to_string(), task_size, task.priority(), fitting));
            }
        }

        if task_info.is_empty() {
            return schedule;
        }

        // Step 1: Get policy-driven task ordering via RL episode.
        // Build task templates and run episode to get priority ordering.
        let mut templates = Vec::new();
        for (id, size, priority, fitting) in &task_info {
            let value = (*priority as f64).max(1.0);
            let total_capacity: f64 = fitting.iter().map(|(s, e)| e - s).sum();
            let urgency_steps = (total_capacity / size).ceil() as u32;
            let deadline = urgency_steps
                .max(1)
                .min(self_mut.config.episode_horizon);

            templates.push(TaskTemplate {
                name: id.clone(),
                value_range: (value, value),
                deadline_range: (deadline, deadline),
                type_requirements: AgentTypeRequirements::new(1, 0, 0),
                collection_radius: Some(self_mut.config.collection_radius),
                max_appearances: Some(1),
                appearances_used: 0,
            });
        }

        // Run RL episode to determine ordering
        let mut env =
            RLEnvironment::with_templates(self_mut.config.clone(), templates, 0);
        env.set_agents(&[(1, AgentType::Young)]);

        let mut obs = env.reset();
        let mut policy_order: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for _ in 0..self_mut.config.episode_horizon {
            let actions = self_mut.policy.select_actions(&obs);
            let result = env.step(actions);

            // Collected instance IDs are "{task_id}_{counter}"; recover original task ID
            for instance_id in &env.task_pool.collected_ids {
                let task_id = instance_id_to_task_id(instance_id);
                if seen.insert(task_id.to_string()) {
                    policy_order.push(task_id.to_string());
                }
            }

            obs = result.observations;
            if result.done {
                break;
            }
        }

        // Build a lookup for task_info
        let task_lookup: std::collections::HashMap<String, (f64, Vec<(f64, f64)>)> = task_info
            .iter()
            .map(|(id, size, _, fitting)| (id.clone(), (*size, fitting.clone())))
            .collect();

        // Step 2: Place tasks in policy order using constraint-respecting placement.
        // Tasks collected by the policy go first, remaining tasks follow in greedy order.
        let mut scheduled_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Phase A: Policy-ordered tasks
        for id in &policy_order {
            if let Some((size, fitting)) = task_lookup.get(id) {
                if let Some(start) = find_earliest_non_overlapping(
                    fitting,
                    *size,
                    horizon_start,
                    horizon_end,
                    &schedule,
                ) {
                    let end = start + size;
                    let interval =
                        Interval::new(Quantity::new(start), Quantity::new(end));
                    let _ = schedule.add(id, interval);
                    scheduled_ids.insert(id.clone());
                }
            }
        }

        // Phase B: Remaining tasks (not selected by policy) in greedy score order
        let mut remaining: Vec<&(String, f64, i32, Vec<(f64, f64)>)> = task_info
            .iter()
            .filter(|(id, _, _, _)| !scheduled_ids.contains(id))
            .collect();
        remaining.sort_by(|a, b| {
            let score_a = greedy_score(a.2, a.3.iter().map(|(s, e)| e - s).sum());
            let score_b = greedy_score(b.2, b.3.iter().map(|(s, e)| e - s).sum());
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (id, size, _, fitting) in remaining {
            if let Some(start) = find_earliest_non_overlapping(
                fitting,
                *size,
                horizon_start,
                horizon_end,
                &schedule,
            ) {
                let end = start + size;
                let interval =
                    Interval::new(Quantity::new(start), Quantity::new(end));
                let _ = schedule.add(id, interval);
            }
        }

        schedule
    }
}

/// Scores a task for greedy fallback selection.
fn greedy_score(priority: i32, remaining_capacity: f64) -> f64 {
    let eps = 1e-6;
    let urgency = 1.0 / (eps + remaining_capacity);
    (priority as f64).max(1.0) * urgency
}

/// Finds the earliest placement start that fits within feasible windows
/// without overlapping already-scheduled intervals.
fn find_earliest_non_overlapping<U: Unit>(
    intervals: &[(f64, f64)],
    size: f64,
    cursor: f64,
    horizon_end: f64,
    schedule: &Schedule<U>,
) -> Option<f64> {
    for &(win_start, win_end) in intervals {
        let effective_start = win_start.max(cursor);
        if effective_start + size > win_end || effective_start + size > horizon_end {
            continue;
        }

        let mut candidate_start = effective_start;
        loop {
            let candidate_end = candidate_start + size;
            if candidate_end > win_end || candidate_end > horizon_end {
                break;
            }

            let query =
                Interval::new(Quantity::new(candidate_start), Quantity::new(candidate_end));
            match schedule.is_free(query) {
                Ok(true) => return Some(candidate_start),
                Ok(false) => {
                    if let Ok(conflicts) = schedule.conflicts_vec(query) {
                        if let Some((_, conflict_iv)) = conflicts.first() {
                            candidate_start = conflict_iv.end().value();
                            continue;
                        }
                    }
                    candidate_start += 1e-6;
                }
                Err(_) => break,
            }
        }
    }
    None
}

/// Extracts the original scheduling task ID from an RL task instance ID.
///
/// Instance IDs are formatted as `{template_name}_{counter}` by [`TaskPool::spawn`].
/// This strips the `_{counter}` suffix to recover the original task name.
fn instance_id_to_task_id(instance_id: &str) -> &str {
    instance_id
        .rsplit_once('_')
        .map(|(prefix, _)| prefix)
        .unwrap_or(instance_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use qtty::Second;

    #[derive(Debug, Clone)]
    struct TestTask {
        name: String,
        size: Quantity<Second>,
        priority: i32,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

        fn name(&self) -> &str {
            &self.name
        }
        fn size(&self) -> Quantity<Second> {
            self.size
        }
        fn priority(&self) -> i32 {
            self.priority
        }
    }

    fn make_task(name: &str, size: f64, priority: i32) -> TestTask {
        TestTask {
            name: name.to_string(),
            size: Quantity::new(size),
            priority,
        }
    }

    #[test]
    fn policy_driven_with_untrained_produces_valid_schedule() {
        let config = RLConfig {
            episode_horizon: 20,
            top_m: 3,
            ..RLConfig::default()
        };
        let policy = NeuralPolicy::new(&config, Device::Cpu);
        let scheduler = PolicyDrivenScheduler::with_policy(policy, config);

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id1 = block.add_task(make_task("t1", 100.0, 10));
        let id2 = block.add_task(make_task("t2", 100.0, 5));

        let mut space = SolutionSpace::new();
        space.add_interval(&id1, Interval::from_f64(0.0, 500.0));
        space.add_interval(&id2, Interval::from_f64(0.0, 500.0));

        let horizon = Interval::from_f64(0.0, 500.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        assert_eq!(schedule.len(), 2);

        // Verify no overlaps
        let iv1 = schedule.get_interval(&id1).unwrap();
        let iv2 = schedule.get_interval(&id2).unwrap();
        let (earlier, later) = if iv1.start().value() < iv2.start().value() {
            (iv1, iv2)
        } else {
            (iv2, iv1)
        };
        assert!(earlier.end().value() <= later.start().value());
    }

    #[test]
    fn policy_driven_fallback_handles_remaining() {
        let config = RLConfig {
            episode_horizon: 5, // Very short — policy may not collect all tasks
            top_m: 2,
            ..RLConfig::default()
        };
        let policy = NeuralPolicy::new(&config, Device::Cpu);
        let scheduler = PolicyDrivenScheduler::with_policy(policy, config);

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let mut ids = Vec::new();
        for i in 0..5 {
            ids.push(block.add_task(make_task(&format!("t{}", i), 50.0, i as i32 + 1)));
        }

        let mut space = SolutionSpace::new();
        for id in &ids {
            space.add_interval(id, Interval::from_f64(0.0, 1000.0));
        }

        let horizon = Interval::from_f64(0.0, 1000.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        // All 5 tasks should fit (5 × 50.0 = 250.0 < 1000.0)
        assert_eq!(schedule.len(), 5);
    }
}
