//! RL-based scheduling algorithm adapter.
//!
//! Bridges the generic [`SchedulingAlgorithm`] interface to the RL environment,
//! enabling RL-driven scheduling within the same pipeline as EST or other algorithms.
//!
//! # Milestone 1 constraints
//!
//! - **Time-only**: spatial displacement (movement/slew) is ignored; all agents and
//!   tasks are placed at the origin. The RL acts purely on time-window assignment.
//! - **Single agent type**: all telescopes/resources are mapped to `AgentType::Young`.
//!   Multi-type requirements are not supported and will cause a panic at construction.
//! - **One resource per task**: coalition semantics are collapsed to a single agent.
//!
//! These constraints will be relaxed in future milestones.

use qtty::{Quantity, Unit};

use crate::algorithms::SchedulingAlgorithm;
use crate::schedule::Schedule;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::{Interval, SolutionSpace};

/// RL-based scheduler that uses time-window assignment via a greedy heuristic.
///
/// For each time step the scheduler advances a cursor through the horizon,
/// scores candidate tasks by priority/value and urgency, and greedily assigns
/// the best task to the earliest feasible interval at or after the cursor.
///
/// # Construction
///
/// ```ignore
/// use virolai::algorithms::rl::scheduler::RLScheduler;
///
/// let scheduler = RLScheduler::new();
/// let schedule = scheduler.schedule(&blocks, &solution_space, horizon);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RLScheduler {
    /// Reserved for future use: number of top candidates to consider at each step.
    _top_k: usize,
}

impl RLScheduler {
    /// Creates a new RL scheduler with default settings.
    pub fn new() -> Self {
        Self { _top_k: 10 }
    }

    /// Creates a new RL scheduler considering `top_k` candidates per step.
    #[allow(dead_code)]
    pub fn with_top_k(top_k: usize) -> Self {
        Self { _top_k: top_k }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchedulingAlgorithm implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<T, U, D, E> SchedulingAlgorithm<T, U, D, E> for RLScheduler
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
        let mut schedule = Schedule::new();

        let horizon_start = horizon.start().value();
        let horizon_end = horizon.end().value();

        // Note: schedule itself tracks placed tasks. We rely on `schedule.contains_task(id)`.

        // Iterative greedy scheduling.
        // Each iteration scans all blocks, queries solution_space directly for each
        // unscheduled task, picks the highest-scoring placeable task, and schedules it.
        loop {
            let mut best_id: Option<String> = None;
            let mut best_start: f64 = f64::MAX;
            let mut best_score: f64 = f64::NEG_INFINITY;
            let mut best_size: f64 = 0.0;

            for block in blocks {
                for (id, task) in block.tasks() {
                    if schedule.contains_task(id) {
                        continue;
                    }

                    let Some(raw_intervals) = solution_space.get_intervals(id) else {
                        continue;
                    };

                    let task_size = task.size_on_axis().value();

                    // Filter intervals that can actually fit this task.
                    let fitting: Vec<(f64, f64)> = raw_intervals
                        .iter()
                        .filter(|iv| iv.duration().value() >= task_size)
                        .map(|iv| (iv.start().value(), iv.end().value()))
                        .collect();

                    let Some(start) = find_earliest_non_overlapping::<U>(
                        &fitting,
                        task_size,
                        horizon_start,
                        horizon_end,
                        &schedule,
                    ) else {
                        continue;
                    };

                    let total_capacity: f64 = fitting.iter().map(|&(s, e)| e - s).sum();
                    let score = greedy_score(task.priority(), total_capacity);

                    // Prefer: (1) highest score, (2) earliest start as tiebreaker.
                    if score > best_score || (score == best_score && start < best_start) {
                        best_score = score;
                        best_start = start;
                        best_size = task_size;
                        best_id = Some(id.to_string());
                    }
                }
            }

            match best_id {
                Some(id) => {
                    let task_end = best_start + best_size;
                    let interval =
                        Interval::new(Quantity::new(best_start), Quantity::new(task_end));

                    // Insert into schedule (should not fail since we checked overlaps).
                    let _ = schedule.add(&id, interval);
                }
                None => break, // No more schedulable tasks.
            }
        }

        schedule
    }
}

/// Scores a task for greedy selection.
///
/// Combines priority with urgency (inverse of remaining feasible capacity).
/// Tasks with less scheduling room get a higher urgency multiplier.
fn greedy_score(priority: i32, remaining_capacity: f64) -> f64 {
    let eps = 1e-6;
    let urgency = 1.0 / (eps + remaining_capacity);
    let value_component = (priority as f64).max(1.0);
    value_component * urgency
}

/// Finds the earliest placement start ≥ `cursor` that:
/// 1. Falls within one of the feasible `intervals`
/// 2. The task (of `size`) fits entirely within that interval
/// 3. Does not overlap any already-scheduled interval
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

        // Try candidate starts within this window. We advance by jumping past
        // conflicting scheduled tasks by querying `schedule` directly.
        let mut candidate_start = effective_start;
        loop {
            let candidate_end = candidate_start + size;
            if candidate_end > win_end || candidate_end > horizon_end {
                break; // Can't fit in this window anymore
            }

            // Build interval in axis units and ask schedule if it's free.
            let query = Interval::new(Quantity::new(candidate_start), Quantity::new(candidate_end));
            match schedule.is_free(query) {
                Ok(true) => return Some(candidate_start),
                Ok(false) => {
                    // There is at least one conflict; find the conflicting task
                    // and jump past its end to continue searching.
                    if let Ok(conflicts) = schedule.conflicts_vec(query) {
                        if let Some((_, conflict_iv)) = conflicts.first() {
                            candidate_start = conflict_iv.end().value();
                            continue;
                        }
                    }
                    // Fallback: advance a tiny epsilon to avoid infinite loop.
                    candidate_start += 1e-6;
                }
                Err(_) => {
                    // On schedule error treat as not placeable in this attempt.
                    break;
                }
            }
        }
    }
    None
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
    fn empty_input_returns_empty_schedule() {
        let scheduler = RLScheduler::new();
        let blocks: Vec<SchedulingBlock<TestTask, Second>> = vec![];
        let space = SolutionSpace::new();
        let horizon = Interval::from_f64(0.0, 1000.0);
        let schedule = scheduler.schedule(&blocks, &space, horizon);
        assert!(schedule.is_empty());
    }

    #[test]
    fn single_task_scheduled_at_earliest() {
        let scheduler = RLScheduler::new();

        let task = make_task("obs1", 100.0, 5);
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id = block.add_task(task);

        let mut space = SolutionSpace::new();
        space.add_interval(&id, Interval::from_f64(200.0, 500.0));

        let horizon = Interval::from_f64(0.0, 1000.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        assert_eq!(schedule.len(), 1);
        let interval = schedule.get_interval(&id).unwrap();
        assert!((interval.start().value() - 200.0).abs() < 1e-6);
        assert!((interval.end().value() - 300.0).abs() < 1e-6);
    }

    #[test]
    fn multiple_tasks_no_overlap() {
        let scheduler = RLScheduler::new();

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id1 = block.add_task(make_task("t1", 100.0, 10));
        let id2 = block.add_task(make_task("t2", 100.0, 5));
        let id3 = block.add_task(make_task("t3", 100.0, 1));

        let mut space = SolutionSpace::new();
        // All tasks can go in the same window
        space.add_interval(&id1, Interval::from_f64(0.0, 500.0));
        space.add_interval(&id2, Interval::from_f64(0.0, 500.0));
        space.add_interval(&id3, Interval::from_f64(0.0, 500.0));

        let horizon = Interval::from_f64(0.0, 500.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        assert_eq!(schedule.len(), 3);

        // Verify no overlaps
        let mut intervals: Vec<_> = [&id1, &id2, &id3]
            .iter()
            .filter_map(|id| schedule.get_interval(id))
            .collect();
        intervals.sort_by(|a, b| a.start().value().partial_cmp(&b.start().value()).unwrap());

        for w in intervals.windows(2) {
            assert!(
                w[0].end().value() <= w[1].start().value(),
                "Overlap detected: {:?} and {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn task_without_feasible_window_is_skipped() {
        let scheduler = RLScheduler::new();

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id1 = block.add_task(make_task("fits", 100.0, 5));
        let id2 = block.add_task(make_task("too_big", 1000.0, 10));

        let mut space = SolutionSpace::new();
        space.add_interval(&id1, Interval::from_f64(0.0, 200.0));
        space.add_interval(&id2, Interval::from_f64(0.0, 200.0)); // too small for 1000s task

        let horizon = Interval::from_f64(0.0, 200.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        assert_eq!(schedule.len(), 1);
        assert!(schedule.get_interval(&id1).is_some());
        assert!(schedule.get_interval(&id2).is_none());
    }

    #[test]
    fn disjoint_windows_used() {
        let scheduler = RLScheduler::new();

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id1 = block.add_task(make_task("t1", 50.0, 5));
        let id2 = block.add_task(make_task("t2", 50.0, 5));

        let mut space = SolutionSpace::new();
        space.add_interval(&id1, Interval::from_f64(0.0, 100.0));
        // t2 only fits in a later window
        space.add_interval(&id2, Interval::from_f64(200.0, 300.0));

        let horizon = Interval::from_f64(0.0, 400.0);
        let schedule = scheduler.schedule(&[block], &space, horizon);

        assert_eq!(schedule.len(), 2);
        let iv1 = schedule.get_interval(&id1).unwrap();
        let iv2 = schedule.get_interval(&id2).unwrap();
        assert!(iv1.end().value() <= iv2.start().value());
    }
}
