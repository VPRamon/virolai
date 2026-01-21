use crate::solution_space::Interval;
use qtty::Unit;

/// Computes the union of two sorted, non-overlapping interval sets.
///
/// # Arguments
///
/// * `a` - First set of sorted, non-overlapping intervals
/// * `b` - Second set of sorted, non-overlapping intervals
///
/// # Returns
///
/// A vector of intervals representing the union, sorted and non-overlapping.
/// Helper function to merge an interval into result, merging with the last one when needed.
fn merge_into<U: Unit>(result: &mut Vec<Interval<U>>, iv: Interval<U>) {
    if let Some(last) = result.last_mut() {
        if last.overlaps(&iv) || last.end().value() == iv.start().value() {
            let new_end = if last.end().value() > iv.end().value() {
                last.end()
            } else {
                iv.end()
            };
            *last = Interval::new(last.start(), new_end);
            return;
        }
    }
    result.push(iv);
}

pub fn compute_union<U: Unit>(a: &[Interval<U>], b: &[Interval<U>]) -> Vec<Interval<U>> {
    // assert a and b are canonical (debug-only)
    debug_assert!(super::assertions::is_canonical(a));
    debug_assert!(super::assertions::is_canonical(b));
    let mut result: Vec<Interval<U>> = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        let ia = &a[i];
        let ib = &b[j];

        if ia.start().value() <= ib.start().value() {
            merge_into(&mut result, *ia);
            i += 1;
        } else {
            merge_into(&mut result, *ib);
            j += 1;
        }
    }

    while i < a.len() {
        merge_into(&mut result, a[i]);
        i += 1;
    }

    while j < b.len() {
        merge_into(&mut result, b[j]);
        j += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution_space::Interval;
    use qtty::Second;

    #[test]
    fn test_compute_union_simple() {
        let a = vec![Interval::<Second>::from_f64(0.0, 50.0)];
        let b = vec![Interval::<Second>::from_f64(100.0, 150.0)];
        let u = compute_union(&a, &b);
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].start().value(), 0.0);
        assert_eq!(u[0].end().value(), 50.0);
        assert_eq!(u[1].start().value(), 100.0);
        assert_eq!(u[1].end().value(), 150.0);
    }

    #[test]
    fn test_compute_union_overlapping() {
        let a = vec![Interval::<Second>::from_f64(0.0, 100.0)];
        let b = vec![Interval::<Second>::from_f64(50.0, 150.0)];
        let u = compute_union(&a, &b);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].start().value(), 0.0);
        assert_eq!(u[0].end().value(), 150.0);
    }

    #[test]
    fn test_compute_union_adjacent() {
        let a = vec![Interval::<Second>::from_f64(0.0, 50.0)];
        let b = vec![Interval::<Second>::from_f64(50.0, 100.0)];
        let u = compute_union(&a, &b);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].start().value(), 0.0);
        assert_eq!(u[0].end().value(), 100.0);
    }
}
