//! Early Starting Time (EST) scheduling algorithm.
//!
//! This algorithm schedules tasks by prioritizing those with the earliest possible start time,
//! considering flexibility and deadline constraints. The algorithm follows these key principles:
//!
//! 1. **Candidate Prioritization**: Tasks are sorted based on:
//!    - Impossible tasks (no valid EST) go last
//!    - Endangered vs Flexible classification based on flexibility threshold
//!    - Cross-kind comparison: flexible tasks may go before endangered if they don't block them
//!    - Same-kind comparison: earlier EST, higher priority, less flexibility, then task ID
//!
//! 2. **Helper Functions** (following the C++ prototype):
//!    - `find_est`: Returns the earliest start time where a task fits in the intersection of
//!      visibility windows with the scheduling horizon
//!    - `find_deadline`: Returns the latest possible start time for a task
//!    - `compute_flexibility`: Sums the ratio of available time to task duration across all
//!      intersecting visibility windows
//!
//! 3. **Scheduling Process**:
//!    - Initialize candidates with metrics (EST, deadline, flexibility)
//!    - Repeatedly schedule the highest-priority candidate
//!    - After each scheduling, move the horizon forward to start after the scheduled task
//!    - Re-compute metrics for remaining candidates with the updated horizon
//!    - Continue until no more schedulable tasks remain
//!
//! # Note on Interval Semantics
//!
//! The implementation uses inclusive interval endpoints `[start, end]`, so a small offset (1 time unit)
//! is added when moving the horizon forward to avoid overlap with the just-scheduled task.
//!
//! # Module Structure
//!
//! This module is organized into the following submodules:
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
    use qtty::Second;

    #[derive(Debug, Clone)]
    struct TestTask {
        id: String,
        name: String,
        size: qtty::Quantity<Second>,
        priority: i32,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> String {
            self.name.clone()
        }

        fn size(&self) -> qtty::Quantity<Second> {
            self.size
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[test]
    fn test_candidate_creation() {
        let task = TestTask {
            id: "1".to_string(),
            name: "Test".to_string(),
            size: qtty::Quantity::new(10.0),
            priority: 5,
        };

        let candidate = Candidate::new(task.clone());
        assert_eq!(candidate.task_id(), "1");
        assert!(candidate.is_impossible());
    }
}
