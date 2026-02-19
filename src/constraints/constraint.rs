//! Core constraint trait for computing valid scheduling intervals.

use crate::solution_space::Interval;
use crate::solution_space::IntervalSet;
use qtty::Unit;
use std::fmt::Debug;

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
/// - Handle empty results gracefully (return empty `IntervalSet`)
pub trait Constraint<U: Unit>: Send + Sync + Debug {
    /// Computes intervals where this constraint is satisfied within `[start, end]`.
    fn compute_intervals(&self, range: Interval<U>) -> IntervalSet<U>;

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

impl<U: Unit + Send + Sync> Constraint<U> for IntervalConstraint<U> {
    fn compute_intervals(&self, range: Interval<U>) -> IntervalSet<U> {
        range
            .intersection(&self.interval())
            .map_or_else(IntervalSet::new, IntervalSet::from)
    }

    fn stringify(&self) -> String {
        self.interval().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::Second;

    fn iv(start: f64, end: f64) -> Interval<Second> {
        Interval::from_f64(start, end)
    }

    #[test]
    fn interval_constraint_new_and_accessor() {
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        assert_eq!(c.interval().start().value(), 10.0);
        assert_eq!(c.interval().end().value(), 50.0);
    }

    #[test]
    fn compute_intervals_range_inside_constraint() {
        // Range [20, 40] inside constraint [10, 50] → [20, 40]
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        let result = c.compute_intervals(iv(20.0, 40.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(20.0, 40.0));
    }

    #[test]
    fn compute_intervals_range_outside_constraint() {
        // Range [60, 80] outside constraint [10, 50] → empty
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        let result = c.compute_intervals(iv(60.0, 80.0));
        assert!(result.is_empty());
    }

    #[test]
    fn compute_intervals_partial_overlap() {
        // Range [30, 80] partial overlap with constraint [10, 50] → [30, 50]
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        let result = c.compute_intervals(iv(30.0, 80.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(30.0, 50.0));
    }

    #[test]
    fn compute_intervals_range_contains_constraint() {
        // Range [0, 100] contains constraint [10, 50] → [10, 50]
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        let result = c.compute_intervals(iv(0.0, 100.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(10.0, 50.0));
    }

    #[test]
    fn stringify_format() {
        let c = IntervalConstraint::new(iv(10.0, 50.0));
        let s = c.stringify();
        assert!(s.contains("10"));
        assert!(s.contains("50"));
    }

    #[test]
    fn print_does_not_panic() {
        let c = IntervalConstraint::new(iv(0.0, 100.0));
        c.print(); // Should not panic
    }
}
