//! Solution space population utilities.

use super::Interval;
use crate::constraints::Constraint;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::Id;
use qtty::{Quantity, Unit};
use std::collections::HashMap;

/// Returns all constraint-derived intervals as a vector for analysis.
///
/// This is a utility function that collects all constraint-computed intervals
/// from multiple blocks into a single vector for analysis purposes.
pub fn collect_intervals<T, U, D, E>(
    blocks: &[SchedulingBlock<T, U, D, E>],
    start: Quantity<U>,
    end: Quantity<U>,
) -> Vec<Interval<U>>
where
    T: Task<U>,
    U: Unit,
    E: petgraph::EdgeType,
{
    let mut intervals = Vec::new();

    for block in blocks {
        for (_id, task) in block.tasks() {
            if let Some(constraint_tree) = task.constraints() {
                let task_intervals = constraint_tree.compute_intervals(Interval::new(start, end));
                intervals.extend(task_intervals);
            }
        }
    }

    intervals
}

impl<U: Unit> super::SolutionSpace<U> {
    /// Populates a solution space from multiple scheduling blocks.
    ///
    /// For each task in each block:
    /// - If the task has constraints, computes valid intervals within the given range
    /// - If the task has no constraints, uses the full range as a single interval
    ///
    /// The solution space maps task IDs to their intervals,
    /// allowing cross-block scheduling with stable task identification.
    ///
    /// # Arguments
    ///
    /// * `blocks` - One or more scheduling blocks to process
    /// * `range` - The scheduling window (interval) to compute valid placements within
    ///
    /// # Returns
    ///
    /// A [`SolutionSpace`] with intervals for all tasks in all blocks
    ///
    /// # Example
    ///
    /// ```ignore
    /// use v_rolai::solution_space::{SolutionSpace, Interval};
    /// use v_rolai::scheduling_block::SchedulingBlock;
    /// use qtty::{Quantity, Second};
    ///
    /// let block = SchedulingBlock::new();
    /// // ... add tasks ...
    ///
    /// let range = Interval::new(
    ///     Quantity::<Second>::new(0.0),
    ///     Quantity::<Second>::new(86400.0)
    /// );
    /// let solution_space = SolutionSpace::populate(&[block], range);
    /// ```
    pub fn populate<T, D, E>(
        blocks: &[crate::scheduling_block::SchedulingBlock<T, U, D, E>],
        range: Interval<U>,
    ) -> Self
    where
        T: crate::scheduling_block::Task<U>,
        E: petgraph::EdgeType,
    {
        let map = blocks
            .iter()
            .flat_map(|block| block.tasks())
            .map(|(id, task)| {
                // Use size_on_axis() to get duration in axis units
                let task_size = task.size_on_axis();
                let intervals = task.constraints().map_or_else(
                    || vec![range],
                    |ct| {
                        ct.compute_intervals(range)
                            .into_iter()
                            .filter(|i| i.duration().value() >= task_size.value())
                            .collect::<Vec<_>>()
                    },
                );
                (id.to_owned(), intervals)
            })
            .collect::<HashMap<Id, Vec<Interval<U>>>>();

        Self::from_hashmap(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ConstraintExpr, IntervalConstraint};
    use crate::scheduling_block::SchedulingBlock;
    use crate::test_utils::TestTask;
    use qtty::Second;

    #[test]
    fn collect_intervals_empty_blocks() {
        let blocks: Vec<SchedulingBlock<TestTask, Second>> = vec![];
        let result = collect_intervals(&blocks, Quantity::new(0.0), Quantity::new(100.0));
        assert!(result.is_empty());
    }

    #[test]
    fn collect_intervals_no_constraints() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        block.add_task(TestTask::new("t", 10.0));

        let result = collect_intervals(&[block], Quantity::new(0.0), Quantity::new(100.0));
        // Tasks without constraints produce no entries in collect_intervals
        assert!(result.is_empty());
    }

    #[test]
    fn collect_intervals_with_constraints() {
        let constraint = IntervalConstraint::new(Interval::from_f64(10.0, 60.0));
        let task = TestTask::new("t", 10.0).with_constraints(ConstraintExpr::leaf(constraint));

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        block.add_task(task);

        let result = collect_intervals(&[block], Quantity::new(0.0), Quantity::new(100.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start().value(), 10.0);
        assert_eq!(result[0].end().value(), 60.0);
    }

    #[test]
    fn populate_filters_by_task_size() {
        // Task size = 50, constraint window = [0, 30] → too small, filtered out
        let constraint = IntervalConstraint::new(Interval::from_f64(0.0, 30.0));
        let task = TestTask::new("t", 50.0).with_constraints(ConstraintExpr::leaf(constraint));

        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task_id = block.add_task(task);

        let range = Interval::from_f64(0.0, 100.0);
        let space = super::super::SolutionSpace::populate(&[block], range);

        let intervals = space.get_intervals(&task_id).unwrap();
        assert!(intervals.is_empty());
    }

    #[test]
    fn populate_multiple_blocks() {
        let mut block1: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id1 = block1.add_task(TestTask::new("t1", 10.0));

        let mut block2: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let id2 = block2.add_task(TestTask::new("t2", 10.0));

        let range = Interval::from_f64(0.0, 100.0);
        let space = super::super::SolutionSpace::populate(&[block1, block2], range);

        assert_eq!(space.count(), 2);
        assert!(space.get_intervals(&id1).is_some());
        assert!(space.get_intervals(&id2).is_some());
    }
}
