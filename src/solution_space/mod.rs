//! Solution space: a collection of valid intervals for task placement.
//!
//! The [`SolutionSpace`] acts as a lookup table that schedulers query to find
//! feasible positions. Users populate it with intervals computed from constraints.

mod interval;
mod populate;
mod space;

pub use interval::Interval;
pub use populate::collect_task_intervals;
pub use space::SolutionSpace;
