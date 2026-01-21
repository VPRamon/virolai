//! Interval set operations for constraint trees.

use crate::solution_space::Interval;
use qtty::Unit;

/// Returns true if `intervals` is canonical: each interval has start <= end,
/// intervals are sorted by start, and they do not overlap (previous end <= next start).
pub fn is_canonical<U: Unit>(intervals: &[Interval<U>]) -> bool {
    intervals.windows(2).all(|w| {
        let prev = &w[0];
        let curr = &w[1];
        !curr.overlaps(prev) && prev.end().value() <= curr.start().value()
    })
}
