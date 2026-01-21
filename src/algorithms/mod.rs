pub mod est;

pub use est::ESTScheduler;

use crate::schedule::Schedule;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::{Interval, SolutionSpace};

/// Algorithm for scheduling tasks from one or more scheduling blocks.
///
/// # Type Parameters
///
/// * `T` - Task type implementing the [`Task`] trait
/// * `U` - Unit of time measurement (e.g., [`qtty::Second`])
/// * `D` - Edge data type (typically `()`)
/// * `E` - Graph edge type (typically [`petgraph::Directed`])
pub trait SchedulingAlgorithm<T, U, D, E>
where
    T: Task<U>,
    U: qtty::Unit,
    E: petgraph::EdgeType,
{
    /// Schedule tasks within the given solution space and scheduling horizon.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Collection of scheduling blocks containing tasks to schedule
    /// * `solution_space` - Valid intervals where each task can be placed
    /// * `horizon` - The time window for scheduling
    ///
    /// # Returns
    ///
    /// A [`Schedule`] containing the scheduled tasks
    fn schedule(
        &self,
        blocks: &[SchedulingBlock<T, U, D, E>],
        solution_space: &SolutionSpace<U>,
        horizon: Interval<U>,
    ) -> Schedule<U>;
}
