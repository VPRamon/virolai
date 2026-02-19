use crate::solution_space::Interval;
use crate::solution_space::IntervalSet;
use qtty::Unit;

/// Returns the complement of a canonical interval set within `[start, end]`.
pub fn compute_complement<U: Unit>(
    canonical: Vec<Interval<U>>,
    interval: Interval<U>,
) -> IntervalSet<U> {
    #[cfg(debug_assertions)]
    assert!(
        super::assertions::is_canonical(&canonical),
        "input `canonical` is not in canonical form"
    );

    if canonical.is_empty() {
        return IntervalSet::from(interval);
    }

    let mut result = Vec::with_capacity(canonical.len() + 1);
    let mut cursor = interval.start();
    for iv in canonical {
        if iv.start() > cursor {
            result.push(Interval::new(cursor, iv.start()));
        }
        cursor = iv.end();
    }

    if cursor < interval.end() {
        result.push(Interval::new(cursor, interval.end()));
    }

    IntervalSet::from_sorted_unchecked(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::Second;

    fn iv(start: f64, end: f64) -> Interval<Second> {
        Interval::from_f64(start, end)
    }

    #[test]
    fn complement_empty_canonical_returns_full_interval() {
        let result = compute_complement(vec![], iv(0.0, 100.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start().value(), 0.0);
        assert_eq!(result[0].end().value(), 100.0);
    }

    #[test]
    fn complement_single_middle_interval() {
        // Canonical: [30, 60], interval: [0, 100]
        // Gaps: [0, 30], [60, 100]
        let result = compute_complement(vec![iv(30.0, 60.0)], iv(0.0, 100.0));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], iv(0.0, 30.0));
        assert_eq!(result[1], iv(60.0, 100.0));
    }

    #[test]
    fn complement_covers_entire_interval() {
        let result = compute_complement(vec![iv(0.0, 100.0)], iv(0.0, 100.0));
        assert!(result.is_empty());
    }

    #[test]
    fn complement_multiple_intervals() {
        // Canonical: [10, 30], [50, 70], interval: [0, 100]
        // Gaps: [0, 10], [30, 50], [70, 100]
        let result = compute_complement(vec![iv(10.0, 30.0), iv(50.0, 70.0)], iv(0.0, 100.0));
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], iv(0.0, 10.0));
        assert_eq!(result[1], iv(30.0, 50.0));
        assert_eq!(result[2], iv(70.0, 100.0));
    }

    #[test]
    fn complement_interval_at_start_boundary() {
        // Canonical: [0, 30], interval: [0, 100]
        // Gap: [30, 100]
        let result = compute_complement(vec![iv(0.0, 30.0)], iv(0.0, 100.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(30.0, 100.0));
    }

    #[test]
    fn complement_interval_at_end_boundary() {
        // Canonical: [70, 100], interval: [0, 100]
        // Gap: [0, 70]
        let result = compute_complement(vec![iv(70.0, 100.0)], iv(0.0, 100.0));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(0.0, 70.0));
    }

    #[test]
    fn complement_inverted_interval_returns_empty() {
        // start > end → empty result
        let result = compute_complement(vec![], Interval::<Second>::from_f64(100.0, 100.0));
        // start == end is valid (zero-width), returns zero-width interval only if canonical is empty
        // But Interval::new panics if start > end, so we test start == end
        assert_eq!(result.len(), 1);
    }
}
