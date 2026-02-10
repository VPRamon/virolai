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
//! ## 3. Scheduling Process (Static Horizon + Cursor)
//!
//! - Initialize candidates with metrics computed against the **static scheduling horizon**
//! - Maintain a separate **cursor** to track scheduling progress
//! - Repeatedly:
//!   1. Schedule the highest-priority candidate at its EST
//!   2. Advance cursor by: `task.end() + task.delay_after()`
//!   3. Partial update: re-compute metrics only for candidates before next endangered task
//!   4. Metrics are always computed against the **static horizon**, not the cursor
//! - Continue until no schedulable tasks remain or cursor exceeds horizon
//!
//! ## 4. Task Delays
//!
//! The algorithm supports inter-task delays via the Task trait:
//! - `delay_after()`: Required delay after this task completes (added to cursor)
//! - `compute_delay_after(previous)`: Delay between two specific tasks (used in ordering)
//!
//! # Differences from Earlier Versions
//!
//! This implementation uses a **static horizon** throughout scheduling, unlike earlier
//! versions that moved the horizon forward. Metrics (EST, deadline, flexibility) are
//! always computed relative to the original full horizon, while a separate cursor
//! variable tracks actual scheduling progress. This matches the C++ core behavior.
//!
//! # Module Structure
//!
//! - [`candidate`] - Task candidate with computed metrics
//! - [`metrics`] - Metric computation functions (EST, deadline, flexibility)
//! - [`ordering`] - Candidate comparison and priority logic
//! - [`engine`] - Core scheduling loop with partial update optimization

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
use engine::{schedule_segment, update_candidates};

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
        let mut candidates: Vec<Candidate<T, U>> = blocks
            .iter()
            .flat_map(|block| {
                block
                    .graph()
                    .node_weights()
                    .map(|task| Candidate::new(task.clone()))
            })
            .collect();

        // Initial update
        update_candidates(
            &mut candidates,
            solution_space,
            horizon,
            self.endangered_threshold,
        );

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
        id: String,
        name: String,
        size: qtty::Quantity<Second>,
        priority: i32,
        delay: qtty::Quantity<Second>,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

        fn id(&self) -> &str {
            &self.id
        }

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
            id: "1".to_string(),
            name: "Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 5,
            delay: qtty::Quantity::new(0.0),
        };

        let candidate = Candidate::new(task.clone());
        assert_eq!(candidate.task_id(), "1");
        assert!(candidate.is_impossible());
    }

    #[test]
    fn test_endangered_threshold_boundary() {
        // Test that flexibility < threshold is endangered, >= threshold is flexible
        // This matches the C++ strict less-than behavior
        
        let task = TestTask {
            id: "boundary".to_string(),
            name: "Boundary Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };

        let mut candidate = Candidate::new(task);
        
        // Set EST so candidate is not impossible
        candidate.est = Some(qtty::Quantity::new(0.0));
        
        let threshold = 5;
        
        // flexibility = 4.9 < 5 → endangered
        candidate.flexibility = qtty::Quantity::new(4.9);
        assert!(candidate.is_endangered(threshold), "4.9 < 5 should be endangered");
        assert!(!candidate.is_flexible(threshold), "4.9 < 5 should not be flexible");
        
        // flexibility = 5.0 >= 5 → flexible (boundary case)
        candidate.flexibility = qtty::Quantity::new(5.0);
        assert!(!candidate.is_endangered(threshold), "5.0 >= 5 should not be endangered");
        assert!(candidate.is_flexible(threshold), "5.0 >= 5 should be flexible");
        
        // flexibility = 5.1 >= 5 → flexible
        candidate.flexibility = qtty::Quantity::new(5.1);
        assert!(!candidate.is_endangered(threshold), "5.1 >= 5 should not be endangered");
        assert!(candidate.is_flexible(threshold), "5.1 >= 5 should be flexible");
        
        // flexibility < 1.0 with no EST → impossible
        candidate.flexibility = qtty::Quantity::new(0.5);
        candidate.est = None;
        assert!(candidate.is_impossible(), "No EST means impossible");
    }

    #[test]
    fn test_task_delay_after() {
        let task_with_delay = TestTask {
            id: "delayed".to_string(),
            name: "Delayed Task".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(5.0), // 5 second delay
        };

        let task_no_delay = TestTask {
            id: "nodelay".to_string(),
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
            id: "static_test".to_string(),
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
        
        let est1 = metrics::compute_est(&task, &solution_space, horizon1);
        let est2 = metrics::compute_est(&task, &solution_space, horizon2);
        
        // EST should differ based on horizon
        assert_eq!(est1, Some(qtty::Quantity::new(0.0)), "EST with horizon1 should be 0");
        assert_eq!(est2, Some(qtty::Quantity::new(100.0)), "EST with horizon2 should be 100");
    }

    #[test]
    fn test_find_next_endangered_index() {
        // Test the partial update optimization helper
        
        let task1 = TestTask {
            id: "1".to_string(),
            name: "Flexible 1".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };
        
        let task2 = TestTask {
            id: "2".to_string(),
            name: "Endangered".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };
        
        let task3 = TestTask {
            id: "3".to_string(),
            name: "Flexible 2".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 0,
            delay: qtty::Quantity::new(0.0),
        };
        
        let mut candidates = vec![
            Candidate::new(task1),
            Candidate::new(task2),
            Candidate::new(task3),
        ];
        
        // Set EST for all candidates so they're not impossible
        candidates[0].est = Some(qtty::Quantity::new(0.0));
        candidates[1].est = Some(qtty::Quantity::new(10.0));
        candidates[2].est = Some(qtty::Quantity::new(20.0));
        
        // Set flexibility values
        candidates[0].flexibility = qtty::Quantity::new(10.0); // Flexible
        candidates[1].flexibility = qtty::Quantity::new(2.0);  // Endangered (< 5)
        candidates[2].flexibility = qtty::Quantity::new(8.0);  // Flexible
        
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
