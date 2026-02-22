//! Candidate comparison and ordering logic.

use crate::scheduling_block::Task;
use qtty::Unit;
use std::cmp::Ordering;

use super::candidate::Candidate;

/// Compares candidates by task ID for deterministic tie-breaking.
#[allow(dead_code)]
pub fn compare_by_id<T, U>(a: &Candidate<T, U>, b: &Candidate<T, U>) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    a.task_id().cmp(b.task_id())
}

/// Compares candidates of the same kind (both endangered or both flexible).
#[allow(dead_code)]
pub fn compare_same_kind<T, U>(a: &Candidate<T, U>, b: &Candidate<T, U>) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    // Earlier EST first
    if let (Some(est_a), Some(est_b)) = (a.est(), b.est()) {
        if est_a.value() != est_b.value() {
            return f64::total_cmp(&est_a.value(), &est_b.value());
        }
    }

    // Higher priority first
    let pa = a.task().priority();
    let pb = b.task().priority();
    if pa != pb {
        return pb.cmp(&pa); // Reversed for higher first
    }

    // Less flexible first
    if a.flexibility().value() != b.flexibility().value() {
        return f64::total_cmp(&a.flexibility().value(), &b.flexibility().value());
    }

    // Tie-breaker: task ID
    compare_by_id(a, b)
}

/// Compares endangered vs flexible candidates.
/// Flexible goes first iff it starts before the endangered and doesn't block it.
///
/// This implementation accounts for inter-task delays to match C++ core behavior.
#[allow(dead_code)]
pub fn compare_endangered_flexible<T, U>(
    endangered: &Candidate<T, U>,
    flexible: &Candidate<T, U>,
) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    if let (Some(est_e), Some(est_f), Some(deadline_e)) =
        (endangered.est(), flexible.est(), endangered.deadline())
    {
        // If endangered starts before or at the same time as flexible, endangered goes first
        if est_e.value() <= est_f.value() {
            return Ordering::Less;
        }

        // Check if scheduling flexible first would block endangered
        // Account for inter-task delay between flexible and endangered
        let flexible_end = est_f + flexible.task().size_on_axis();
        let required_gap = endangered.task().compute_gap_after(flexible.task());
        let endangered_start_after_flexible = flexible_end + required_gap;

        // If endangered can still start before its deadline, flexible can go first
        if endangered_start_after_flexible.value() <= deadline_e.value() {
            return Ordering::Greater; // Flexible goes first
        }
    }

    // Default: endangered goes first
    Ordering::Less
}

/// Main comparison function for sorting candidates.
#[allow(dead_code)]
pub fn compare_candidates<T, U>(
    a: &Candidate<T, U>,
    b: &Candidate<T, U>,
    endangered_threshold: u32,
) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    // Impossible always last
    match (a.is_impossible(), b.is_impossible()) {
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (true, true) => return compare_by_id(a, b),
        (false, false) => {}
    }

    // Handle endangered vs flexible
    let a_endangered = a.is_endangered(endangered_threshold);
    let b_endangered = b.is_endangered(endangered_threshold);
    let a_flexible = a.is_flexible(endangered_threshold);
    let b_flexible = b.is_flexible(endangered_threshold);

    if a_endangered && b_flexible {
        return compare_endangered_flexible(a, b);
    }
    if a_flexible && b_endangered {
        return compare_endangered_flexible(b, a).reverse();
    }

    // Same kind
    compare_same_kind(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTask;
    use qtty::Second;
    use std::cmp::Ordering;

    fn make_candidate(
        id: &str,
        size: f64,
        priority: i32,
        est: Option<f64>,
        deadline: Option<f64>,
        flexibility: f64,
        delay: f64,
    ) -> Candidate<TestTask, Second> {
        let task = TestTask::new(id, size)
            .with_priority(priority)
            .with_delay(delay);
        let mut c = Candidate::new(task, id);
        c.est = est.map(qtty::Quantity::new);
        c.deadline = deadline.map(qtty::Quantity::new);
        c.flexibility = qtty::Quantity::new(flexibility);
        c
    }

    // ── compare_by_id ─────────────────────────────────────────────────

    #[test]
    fn compare_by_id_lexicographic() {
        let a = make_candidate("alpha", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        let b = make_candidate("beta", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        assert_eq!(compare_by_id(&a, &b), Ordering::Less);
        assert_eq!(compare_by_id(&b, &a), Ordering::Greater);
        assert_eq!(compare_by_id(&a, &a), Ordering::Equal);
    }

    // ── compare_same_kind ─────────────────────────────────────────────

    #[test]
    fn same_kind_earlier_est_first() {
        let a = make_candidate("a", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        let b = make_candidate("b", 10.0, 0, Some(10.0), None, 5.0, 0.0);
        assert_eq!(compare_same_kind(&a, &b), Ordering::Less);
    }

    #[test]
    fn same_kind_higher_priority_first() {
        let a = make_candidate("a", 10.0, 5, Some(0.0), None, 5.0, 0.0);
        let b = make_candidate("b", 10.0, 3, Some(0.0), None, 5.0, 0.0);
        assert_eq!(compare_same_kind(&a, &b), Ordering::Less); // Higher priority (5) first
    }

    #[test]
    fn same_kind_less_flexible_first() {
        let a = make_candidate("a", 10.0, 0, Some(0.0), None, 2.0, 0.0);
        let b = make_candidate("b", 10.0, 0, Some(0.0), None, 8.0, 0.0);
        assert_eq!(compare_same_kind(&a, &b), Ordering::Less); // Less flexible first
    }

    #[test]
    fn same_kind_tiebreak_by_id() {
        let a = make_candidate("aaa", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        let b = make_candidate("zzz", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        assert_eq!(compare_same_kind(&a, &b), Ordering::Less);
    }

    // ── compare_endangered_flexible ───────────────────────────────────

    #[test]
    fn endangered_first_when_est_earlier_or_equal() {
        let endangered = make_candidate("e", 10.0, 0, Some(0.0), Some(50.0), 2.0, 0.0);
        let flexible = make_candidate("f", 10.0, 0, Some(5.0), None, 10.0, 0.0);
        // est_e <= est_f → endangered goes first
        assert_eq!(
            compare_endangered_flexible(&endangered, &flexible),
            Ordering::Less
        );
    }

    #[test]
    fn flexible_first_when_fits_before_deadline() {
        // Endangered starts at 30, flexible starts at 0, flexible ends at 10
        // Endangered deadline = 50, so flexible finishing at 10 is fine
        let endangered = make_candidate("e", 10.0, 0, Some(30.0), Some(50.0), 2.0, 0.0);
        let flexible = make_candidate("f", 10.0, 0, Some(0.0), None, 10.0, 0.0);
        assert_eq!(
            compare_endangered_flexible(&endangered, &flexible),
            Ordering::Greater
        );
    }

    #[test]
    fn endangered_first_when_flexible_would_block() {
        // Endangered starts at 15, flexible starts at 0, flexible ends at 10
        // Endangered deadline = 12 — flexible end (10) + delay (0) = 10, but deadline = 12
        // Actually flexible_end + delay = 10 <= 12, so flexible goes first
        // Let's make deadline = 5 to block
        let endangered = make_candidate("e", 10.0, 0, Some(15.0), Some(5.0), 2.0, 0.0);
        let flexible = make_candidate("f", 10.0, 0, Some(0.0), None, 10.0, 0.0);
        // flexible end = 0 + 10 = 10 > deadline 5 → endangered first
        assert_eq!(
            compare_endangered_flexible(&endangered, &flexible),
            Ordering::Less
        );
    }

    #[test]
    fn endangered_default_with_no_est() {
        let endangered = make_candidate("e", 10.0, 0, None, None, 2.0, 0.0);
        let flexible = make_candidate("f", 10.0, 0, Some(0.0), None, 10.0, 0.0);
        assert_eq!(
            compare_endangered_flexible(&endangered, &flexible),
            Ordering::Less
        );
    }

    // ── compare_candidates ────────────────────────────────────────────

    #[test]
    fn impossible_always_last() {
        let possible = make_candidate("a", 10.0, 0, Some(0.0), None, 5.0, 0.0);
        let impossible = make_candidate("b", 10.0, 0, None, None, 0.0, 0.0);
        assert_eq!(
            compare_candidates(&possible, &impossible, 5),
            Ordering::Less
        );
        assert_eq!(
            compare_candidates(&impossible, &possible, 5),
            Ordering::Greater
        );
    }

    #[test]
    fn two_impossible_ordered_by_id() {
        let a = make_candidate("aaa", 10.0, 0, None, None, 0.0, 0.0);
        let b = make_candidate("zzz", 10.0, 0, None, None, 0.0, 0.0);
        assert_eq!(compare_candidates(&a, &b, 5), Ordering::Less);
    }

    #[test]
    fn endangered_vs_flexible_cross_kind() {
        let endangered = make_candidate("e", 10.0, 0, Some(0.0), Some(50.0), 2.0, 0.0);
        let flexible = make_candidate("f", 10.0, 0, Some(20.0), None, 10.0, 0.0);
        // est_e = 0 <= est_f = 20, so endangered goes first
        let result = compare_candidates(&endangered, &flexible, 5);
        assert_eq!(result, Ordering::Less);
    }

    #[test]
    fn same_kind_both_flexible() {
        let a = make_candidate("a", 10.0, 0, Some(0.0), None, 10.0, 0.0);
        let b = make_candidate("b", 10.0, 0, Some(5.0), None, 10.0, 0.0);
        // Both flexible, same kind → earlier EST first
        assert_eq!(compare_candidates(&a, &b, 5), Ordering::Less);
    }

    #[test]
    fn same_kind_both_endangered() {
        let a = make_candidate("a", 10.0, 0, Some(0.0), Some(20.0), 2.0, 0.0);
        let b = make_candidate("b", 10.0, 0, Some(5.0), Some(25.0), 3.0, 0.0);
        // Both endangered, same kind → earlier EST first
        assert_eq!(compare_candidates(&a, &b, 5), Ordering::Less);
    }
}
