use crate::solution_space::Interval;
use qtty::Unit;

/// Returns the complement of a canonical interval set within `[start, end]`.
pub fn compute_complement<U: Unit>(
    canonical: Vec<Interval<U>>,
    interval: Interval<U>,
) -> Vec<Interval<U>> {
    // TODO: assert a and b are canonical
    if interval.start().value() > interval.end().value() {
        return Vec::new();
    }
    if canonical.is_empty() {
        return vec![Interval::new(interval.start(), interval.end())];
    }

    let mut result = Vec::new();
    let mut cursor = interval.start();
    for iv in canonical {
        if iv.start().value() > cursor.value() {
            result.push(Interval::new(cursor, iv.start()));
        }
        // advance cursor to the end of this interval
        cursor = iv.end();
        // continue to next canonical interval to find further gaps
    }

    if cursor.value() < interval.end().value() {
        result.push(Interval::new(cursor, interval.end()));
    }

    result
}
