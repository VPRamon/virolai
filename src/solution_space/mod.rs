//! Solution space: a collection of valid intervals for entity placement.
//!
//! The [`SolutionSpaceND`] type is parametrised by an [`Axes`] tuple that
//! encodes the product of physical dimensions.  For the common 1-D case the
//! type alias `SolutionSpace<U> = SolutionSpaceND<(U,)>` provides a drop-in
//! replacement for the previous concrete type.

mod axes;
mod interval;
mod interval_set;
mod populate;
mod space;

pub use axes::{Axes, Region2, Region3, Region4};
pub use interval::Interval;
pub use interval_set::IntervalSet;
pub use populate::collect_intervals;
pub use space::{SolutionSpace, SolutionSpaceND};
