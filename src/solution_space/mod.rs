//! Solution space: a collection of valid intervals for entity placement.
//!
//! The [`SolutionSpace`] acts as a lookup table that schedulers query to find
//! feasible positions. Can be used for both tasks and resources (instruments).
//! Users populate it with intervals computed from constraints.

mod interval;
mod interval_set;
mod populate;
mod space;

pub use interval::Interval;
pub use interval_set::IntervalSet;
pub use populate::collect_intervals;
pub use space::SolutionSpace;
