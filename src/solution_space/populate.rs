//! Solution space population utilities.

use super::Interval;
use crate::scheduling_block::{SchedulingBlock, Task};
use qtty::{Quantity, Unit};

/// Returns all constraint-derived intervals as a vector for analysis.
///
/// This is a utility function that collects all constraint-computed intervals
/// from multiple blocks into a single vector for analysis purposes.
pub fn collect_task_intervals<T, U, D, E>(
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
        for node_idx in block.graph().node_indices() {
            if let Some(task) = block.get_task(node_idx) {
                if let Some(constraint_tree) = task.constraints() {
                    let task_intervals =
                        constraint_tree.compute_intervals(Interval::new(start, end));
                    intervals.extend(task_intervals);
                }
            }
        }
    }

    intervals
}
