//! Core constraint trait for computing valid scheduling intervals.

use crate::solution_space::Interval;
use qtty::Unit;
use std::fmt::Debug;
use std::vec;

/// Computes intervals where a scheduling condition is satisfied.
///
/// Constraints compose via combinators ([`ConstraintNode`](crate::constraints::ConstraintNode))
/// to form trees representing complex AND/OR logic.
///
/// # Contract
///
/// Implementations should:
/// - Return non-overlapping, sorted intervals within `[start, end]`
/// - Be deterministic for identical inputs
/// - Handle empty results gracefully (return empty `Vec`)
pub trait Constraint<U: Unit>: Send + Sync + Debug {
    /// Computes intervals where this constraint is satisfied within `[start, end]`.
    fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>>;

    /// Returns a string representation of this constraint.
    fn stringify(&self) -> String;

    /// Prints this constraint to stdout.
    fn print(&self) {
        println!("{}", self.stringify());
    }
}

/// A fixed-window constraint that allows scheduling only within `[allowed_start, allowed_end]`.
#[derive(Debug, Clone, Copy)]
pub struct IntervalConstraint<U: Unit + Send + Sync>(Interval<U>);

impl<U: Unit + Send + Sync> IntervalConstraint<U> {
    pub const fn new(interval: Interval<U>) -> Self {
        Self(interval)
    }

    pub const fn interval(&self) -> Interval<U> {
        self.0
    }
}

// TODO: Implement operators <, <=, >, >= for QTTY
impl<U: Unit + Send + Sync> Constraint<U> for IntervalConstraint<U> {
    fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>> {
        range
            .intersection(&self.interval())
            .map_or_else(Vec::new, |intersection| vec![intersection])
    }

    fn stringify(&self) -> String {
        self.interval().to_string()
    }
}
