pub mod est;
pub mod rl;

pub use est::ESTScheduler;
#[cfg(feature = "rl-nn")]
pub use rl::policy_scheduler::RLScheduler;

use std::collections::HashMap;

use crate::schedule::Schedule;
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::{Interval, SolutionSpace};
use crate::Id;

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

/// Algorithm for scheduling tasks across multiple resources.
///
/// Each resource has its own pre-computed solution space. The algorithm returns
/// one schedule per resource.
///
/// # Type Parameters
///
/// Same as [`SchedulingAlgorithm`].
pub trait MultiResourceAlgorithm<T, U, D, E>
where
    T: Task<U>,
    U: qtty::Unit,
    E: petgraph::EdgeType,
{
    /// Schedule tasks across multiple resources.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Collection of scheduling blocks containing tasks to schedule.
    ///   Tasks may appear in multiple resource solution spaces if they are eligible
    ///   for multiple resources (based on resource constraints).
    /// * `resource_spaces` - Map from resource ID to the pre-computed solution space
    ///   for that resource. Each solution space only contains task entries that are
    ///   eligible for that resource.
    /// * `horizon` - The scheduling window.
    ///
    /// # Returns
    ///
    /// A map from resource ID to the schedule produced for that resource.
    fn schedule_multi(
        &self,
        blocks: &[SchedulingBlock<T, U, D, E>],
        resource_spaces: &HashMap<Id, SolutionSpace<U>>,
        horizon: Interval<U>,
    ) -> HashMap<Id, Schedule<U>>;
}

/// Adapter that runs a single-resource [`SchedulingAlgorithm`] independently per resource.
///
/// This is the simplest multi-resource strategy: each resource gets its own schedule
/// computed in isolation. Tasks may be scheduled on multiple resources if they appear
/// in multiple solution spaces (no deduplication).
///
/// # Example
///
/// ```ignore
/// use virolai::algorithms::{ESTScheduler, IndependentScheduler, MultiResourceAlgorithm};
///
/// let multi = IndependentScheduler::new(ESTScheduler::new(100));
/// let schedules = multi.schedule_multi(&blocks, &resource_spaces, horizon);
/// ```
pub struct IndependentScheduler<A> {
    inner: A,
}

impl<A> IndependentScheduler<A> {
    /// Wraps a single-resource algorithm for independent per-resource scheduling.
    pub fn new(algorithm: A) -> Self {
        Self { inner: algorithm }
    }
}

impl<A, T, U, D, E> MultiResourceAlgorithm<T, U, D, E> for IndependentScheduler<A>
where
    A: SchedulingAlgorithm<T, U, D, E>,
    T: Task<U>,
    U: qtty::Unit,
    E: petgraph::EdgeType,
{
    fn schedule_multi(
        &self,
        blocks: &[SchedulingBlock<T, U, D, E>],
        resource_spaces: &HashMap<Id, SolutionSpace<U>>,
        horizon: Interval<U>,
    ) -> HashMap<Id, Schedule<U>> {
        resource_spaces
            .iter()
            .map(|(resource_id, space)| {
                let schedule = self.inner.schedule(blocks, space, horizon);
                (resource_id.clone(), schedule)
            })
            .collect()
    }
}
