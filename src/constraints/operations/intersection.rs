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

    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        let ia = &a[i];
        let ib = &b[j];

        if ia.overlaps(ib) {
            result.push(Interval::new(
                crate::constraints::quantity_max(ia.start(), ib.start()),
                crate::constraints::quantity_min(ia.end(), ib.end()),
            ));
        }

        match ia.end().partial_cmp(&ib.end()) {
            Some(std::cmp::Ordering::Less) => i += 1,
            Some(std::cmp::Ordering::Greater) => j += 1,
            _ => {
                i += 1;
                j += 1;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::Second;

    fn iv(start: f64, end: f64) -> Interval<Second> {
        Interval::from_f64(start, end)
    }

    #[test]
    fn intersection_disjoint_sets() {
        let a = vec![iv(0.0, 10.0)];
        let b = vec![iv(20.0, 30.0)];
        let result = compute_intersection(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn intersection_fully_overlapping() {
        let a = vec![iv(0.0, 100.0)];
        let b = vec![iv(20.0, 80.0)];
        let result = compute_intersection(&a, &b);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(20.0, 80.0));
    }

    #[test]
    fn intersection_partial_overlap() {
        let a = vec![iv(0.0, 50.0)];
        let b = vec![iv(30.0, 80.0)];
        let result = compute_intersection(&a, &b);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(30.0, 50.0));
    }

    #[test]
    fn intersection_both_empty() {
        let a: Vec<Interval<Second>> = vec![];
        let b: Vec<Interval<Second>> = vec![];
        let result = compute_intersection(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn intersection_one_empty() {
        let a = vec![iv(0.0, 50.0)];
        let b: Vec<Interval<Second>> = vec![];
        let result = compute_intersection(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn intersection_identical_intervals() {
        let a = vec![iv(10.0, 50.0)];
        let b = vec![iv(10.0, 50.0)];
        let result = compute_intersection(&a, &b);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(10.0, 50.0));
    }

    #[test]
    fn intersection_multiple_intervals() {
        // A: [0, 30], [50, 80]
        // B: [10, 60]
        // Result: [10, 30], [50, 60]
        let a = vec![iv(0.0, 30.0), iv(50.0, 80.0)];
        let b = vec![iv(10.0, 60.0)];
        let result = compute_intersection(&a, &b);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], iv(10.0, 30.0));
        assert_eq!(result[1], iv(50.0, 60.0));
    }

    #[test]
    fn intersection_touching_endpoints() {
        // A ends at 50, B starts at 50 → touching point
        let a = vec![iv(0.0, 50.0)];
        let b = vec![iv(50.0, 100.0)];
        let result = compute_intersection(&a, &b);
        // They overlap at point 50 (inclusive endpoints)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(50.0, 50.0));
    }
}
