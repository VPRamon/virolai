//! Early Starting Time (EST) scheduling algorithm.
//!
//! This algorithm schedules tasks by prioritizing those with the earliest possible start time,
//! considering flexibility and deadline constraints. The implementation matches the C++ core
//! EST algorithm behavior precisely.
//!
//! # Key Principles
//!
//! ## 1. Candidate Prioritization
//!
//! Tasks are sorted based on:
//! - Impossible tasks (no valid EST) go last
//! - Endangered vs Flexible classification based on flexibility threshold
//!   - Endangered: flexibility < threshold (strict less-than)
//!   - Flexible: flexibility >= threshold
//! - Cross-kind comparison: flexible tasks may go before endangered if they don't block them
//!   (accounting for inter-task delays)
//! - Same-kind comparison: earlier EST, higher priority, less flexibility, then task ID
//!
//! ## 2. Metric Functions
//!
//! - `compute_est`: Earliest start time where a task fits in visibility windows ∩ horizon
//! - `compute_deadline`: Latest possible start time for a task
//! - `compute_flexibility`: Sum of (available_time / task_duration) across windows
//!   - flexibility < 1.0 → impossible to schedule
//!   - flexibility < threshold → endangered
//!   - flexibility >= threshold → flexible
//!
//! ## 3. Scheduling Process (Rolling Horizon + Cursor)
//!
//! - Maintain a separate **cursor** to track scheduling progress
//! - Repeatedly:
//!   1. Schedule the highest-priority candidate at its EST
//!   2. Advance cursor by: `task.end() + task.delay_after()`
//!   3. Re-compute candidate metrics on the remaining horizon `[cursor, end]`
//! - Continue until no schedulable tasks remain or cursor exceeds horizon
//!
//! ## 4. Task Delays
//!
//! The algorithm supports inter-task delays via the Task trait:
//! - `delay_after()`: Required delay after this task completes (added to cursor)
//! - `compute_delay_after(previous)`: Delay between two specific tasks (used in ordering)
//!
//! # Module Structure
//!
//! - [`candidate`] - Task candidate with computed metrics
//! - [`metrics`] - Metric computation functions (EST, deadline, flexibility)
//! - [`ordering`] - Candidate comparison and priority logic
//! - [`engine`] - Core scheduling loop and candidate updates

mod candidate;
mod engine;
mod metrics;
mod ordering;

use crate::schedule::Schedule;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::Interval;
use crate::solution_space::SolutionSpace;
use qtty::Unit;

use candidate::Candidate;
use engine::schedule_segment;

/// Early Starting Time scheduler.
pub struct ESTScheduler {
    endangered_threshold: u32,
}

impl ESTScheduler {
    /// Creates a new EST scheduler with the given endangered threshold.
    ///
    /// # Arguments
    ///
    /// * `endangered_threshold` - Tasks with flexibility <= this value are considered endangered
    pub fn new(endangered_threshold: u32) -> Self {
        Self {
            endangered_threshold,
        }
    }
}

impl Default for ESTScheduler {
    /// Creates a new EST scheduler with default threshold of 1.
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T, U, D, E> crate::algorithms::SchedulingAlgorithm<T, U, D, E> for ESTScheduler
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

        // Collect all tasks from all blocks
        let candidates: Vec<Candidate<T, U>> = blocks
            .iter()
            .flat_map(|block| {
                block.tasks().map(|(id, task)| Candidate::new(task.clone(), id))
            })
            .collect();

        // Schedule
        schedule_segment(
            &mut schedule,
            candidates,
            solution_space,
            horizon,
            self.endangered_threshold,
        );

        schedule
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use engine::find_next_endangered_index;
    use qtty::Second;

    #[derive(Debug, Clone)]
    struct TestTask {
        name: String,
        size: qtty::Quantity<Second>,
        priority: i32,
        delay: qtty::Quantity<Second>,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

        fn name(&self) -> &str {
            &self.name
        }

        fn size(&self) -> qtty::Quantity<Second> {
            self.size
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn delay_after(&self) -> qtty::Quantity<Second> {
            self.delay
        }
    }

    #[test]
    fn test_candidate_creation() {
        let task = TestTask {
            name: "Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 5,
            delay: qtty::Quantity::new(0.0),
        };

        let candidate = Candidate::new(task.clone(), "test-id");
        assert_eq!(candidate.task_id(), "test-id");
        assert!(candidate.is_impossible());
    }

    #[test]
    fn test_endangered_threshold_boundary() {
        // Test that flexibility < threshold is endangered, >= threshold is flexible
        // This matches the C++ strict less-than behavior

        let task = TestTask {
            name: "Boundary Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        let mut candidate =Candidate::new(task, "boundary");

        // Set EST so candidate is not impossible
        candidate.est = Some(qtty::Quantity::new(0.0));

        let threshold = 5;

        // flexibility = 4.9 < 5 → endangered
        candidate.flexibility = qtty::Quantity::new(4.9);
        assert!(
            candidate.is_endangered(threshold),
            "4.9 < 5 should be endangered"
        );
        assert!(
            !candidate.is_flexible(threshold),
            "4.9 < 5 should not be flexible"
        );

        // flexibility = 5.0 >= 5 → flexible (boundary case)
        candidate.flexibility = qtty::Quantity::new(5.0);
        assert!(
            !candidate.is_endangered(threshold),
            "5.0 >= 5 should not be endangered"
        );
        assert!(
            candidate.is_flexible(threshold),
            "5.0 >= 5 should be flexible"
        );

        // flexibility = 5.1 >= 5 → flexible
        candidate.flexibility = qtty::Quantity::new(5.1);
        assert!(
            !candidate.is_endangered(threshold),
            "5.1 >= 5 should not be endangered"
        );
        assert!(
            candidate.is_flexible(threshold),
            "5.1 >= 5 should be flexible"
        );

        // flexibility < 1.0 with no EST → impossible
        candidate.flexibility = qtty::Quantity::new(0.5);
        candidate.est = None;
        assert!(candidate.is_impossible(), "No EST means impossible");
    }

    #[test]
    fn test_task_delay_after() {
        let task_with_delay = TestTask {
            name: "Delayed Task".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(5.0), // 5 second delay
        };

        let task_no_delay = TestTask {
            name: "No Delay Task".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0), // No delay
        };

        assert_eq!(task_with_delay.delay_after().value(), 5.0);
        assert_eq!(task_no_delay.delay_after().value(), 0.0);
    }

    #[test]
    fn test_static_horizon_in_metrics() {
        // This test verifies that metrics are computed against a static horizon
        // We can't directly test the scheduling loop here, but we can verify
        // that the metric functions work correctly with a fixed horizon

        use crate::solution_space::{Interval, SolutionSpace};

        let task = TestTask {
            name: "Static Horizon Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        // Create a solution space with visibility windows
        let mut solution_space = SolutionSpace::<Second>::new();
        solution_space.set_intervals(
            "static_test".to_string(),
            vec![
                Interval::new(qtty::Quantity::new(0.0), qtty::Quantity::new(50.0)),
                Interval::new(qtty::Quantity::new(100.0), qtty::Quantity::new(150.0)),
            ],
        );

        // Test with different horizons - metrics should reflect the horizon used
        let horizon1 = Interval::new(qtty::Quantity::new(0.0), qtty::Quantity::new(100.0));
        let horizon2 = Interval::new(qtty::Quantity::new(50.0), qtty::Quantity::new(200.0));

        let est1 = metrics::compute_est(&task, "static_test", &solution_space, horizon1);
        let est2 = metrics::compute_est(&task, "static_test", &solution_space, horizon2);

        // EST should differ based on horizon
        assert_eq!(
            est1,
            Some(qtty::Quantity::new(0.0)),
            "EST with horizon1 should be 0"
        );
        assert_eq!(
            est2,
            Some(qtty::Quantity::new(100.0)),
            "EST with horizon2 should be 100"
        );
    }

    #[test]
    fn test_schedule_recomputes_est_from_cursor() {
        use crate::algorithms::SchedulingAlgorithm;
        use crate::solution_space::{Interval, SolutionSpace};

        let task1 = TestTask {
            name: "Task 1".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 1,
            delay: qtty::Quantity::new(0.0),
        };
        let task2 = TestTask {
            name: "Task 2".to_string(),
            size:qtty::Quantity::new(10.0),
            priority: 1,
            delay: qtty::Quantity::new(0.0),
        };

        // Both tasks share one long window. Without cursor-aware recomputation,
        // both get EST=0 and the second task is dropped due to overlap.
        
        let horizon = Interval::new(qtty::Quantity::new(0.0), qtty::Quantity::new(100.0));

        let mut block1: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task1_id = block1.add_task(task1);
        let mut block2: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task2_id = block2.add_task(task2);

        let mut solution_space = SolutionSpace::<Second>::new();
        solution_space.set_intervals(
            task1_id,
            vec![Interval::new(
                qtty::Quantity::new(0.0),
                qtty::Quantity::new(100.0),
            )],
        );
        solution_space.set_intervals(
            task2_id,
            vec![Interval::new(
                qtty::Quantity::new(0.0),
                qtty::Quantity::new(100.0),
            )],
        );

        let scheduler = ESTScheduler::new(5);
        let schedule = scheduler.schedule(&[block1, block2], &solution_space, horizon);

        assert_eq!(schedule.len(), 2, "Both tasks should be scheduled");

        let entries: Vec<_> = schedule.iter().collect();
        let first = entries[0].1;
        let second = entries[1].1;

        assert!((first.start().value() - 0.0).abs() < 1e-9);
        assert!(
            second.start().value() > first.end().value(),
            "Second task should start strictly after the first ends"
        );
        assert!(
            second.start().value() - first.end().value() < 1e-3,
            "Gap should only be the scheduler epsilon"
        );
    }

    #[test]
    fn test_find_next_endangered_index() {
        // Test the partial update optimization helper

        let task1 = TestTask {
            name: "Flexible 1".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        let task2 = TestTask {
            name: "Endangered".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        let task3 = TestTask {
            name: "Flexible 2".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        let mut candidates = vec![
            Candidate::new(task1, "1"),
            Candidate::new(task2, "2"),
            Candidate::new(task3, "3"),
        ];

        // Set EST for all candidates so they're not impossible
        candidates[0].est = Some(qtty::Quantity::new(0.0));
        candidates[1].est = Some(qtty::Quantity::new(10.0));
        candidates[2].est = Some(qtty::Quantity::new(20.0));

        // Set flexibility values
        candidates[0].flexibility = qtty::Quantity::new(10.0); // Flexible
        candidates[1].flexibility = qtty::Quantity::new(2.0); // Endangered (< 5)
        candidates[2].flexibility = qtty::Quantity::new(8.0); // Flexible

        let threshold = 5;

        // Find next endangered - should be at index 1
        let next_endangered = find_next_endangered_index(&candidates, threshold);
        assert_eq!(next_endangered, 1, "Next endangered should be at index 1");

        // If no endangered tasks, should return len()
        candidates[1].flexibility = qtty::Quantity::new(10.0); // Make all flexible
        let next_endangered = find_next_endangered_index(&candidates, threshold);
        assert_eq!(next_endangered, 3, "No endangered should return len()");
    }
}
