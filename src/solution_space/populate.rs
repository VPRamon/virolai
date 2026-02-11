//! Solution space population utilities.

use super::Interval;
use crate::Id;
use crate::constraints::Constraint;
use crate::scheduling_block::{SchedulingBlock, Task};
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
                let task_intervals =
                    constraint_tree.compute_intervals(Interval::new(start, end));
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
                            .collect()
                    },
                );
                (id.to_owned(), intervals)
            })
            .collect::<HashMap<Id, Vec<Interval<U>>>>();

        Self::from_hashmap(map)
    }
}