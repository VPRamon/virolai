use crate::solution_space::Interval;
use qtty::Unit;

/// Computes the intersection of two sorted interval sets.
///
/// # Arguments
///
/// * `a` - First set of sorted, non-overlapping intervals
/// * `b` - Second set of sorted, non-overlapping intervals
///
/// # Returns
///
/// A vector of intervals representing the intersection, sorted and non-overlapping.
pub fn compute_intersection<U: Unit>(a: &[Interval<U>], b: &[Interval<U>]) -> Vec<Interval<U>> {
    // assert a and b are canonical (debug-only)
    debug_assert!(super::assertions::is_canonical(a));
    debug_assert!(super::assertions::is_canonical(b));
    let mut result = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        let ia = &a[i];
        let ib = &b[j];

        if ia.overlaps(ib) {
            let start = if ia.start().value() > ib.start().value() {
                ia.start()
            } else {
                ib.start()
            };
            let end = if ia.end().value() < ib.end().value() {
                ia.end()
            } else {
                ib.end()
            };
            result.push(Interval::new(start, end));
        }

        let a_end = ia.end().value();
        let b_end = ib.end().value();

        if a_end < b_end {
            i += 1;
        } else if b_end < a_end {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }

    result.shrink_to_fit();
    result
}
