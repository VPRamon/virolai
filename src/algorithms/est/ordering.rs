//! Candidate comparison and ordering logic.

use crate::scheduling_block::Task;
use qtty::Unit;
use std::cmp::Ordering;

use super::candidate::Candidate;

/// Compares candidates by task ID for deterministic tie-breaking.
pub fn compare_by_id<T, U>(a: &Candidate<T, U>, b: &Candidate<T, U>) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    a.task_id().cmp(b.task_id())
}

/// Compares candidates of the same kind (both endangered or both flexible).
pub fn compare_same_kind<T, U>(a: &Candidate<T, U>, b: &Candidate<T, U>) -> Ordering
where
    T: Task<U>,
    U: Unit,
{
    // Earlier EST first
    if let (Some(est_a), Some(est_b)) = (a.est(), b.est()) {
        if est_a.value() != est_b.value() {
            return est_a
                .value()
                .partial_cmp(&est_b.value())
                .unwrap_or(Ordering::Equal);
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
        return a
            .flexibility()
            .value()
            .partial_cmp(&b.flexibility().value())
            .unwrap_or(Ordering::Equal);
    }

    // Tie-breaker: task ID
    compare_by_id(a, b)
}

/// Compares endangered vs flexible candidates.
/// Flexible goes first iff it starts before the endangered and doesn't block it.
///
/// This implementation accounts for inter-task delays to match C++ core behavior.
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
        let required_delay = endangered.task().compute_delay_after(flexible.task());
        let endangered_start_after_flexible = flexible_end + required_delay;

        // If endangered can still start before its deadline, flexible can go first
        if endangered_start_after_flexible.value() <= deadline_e.value() {
            return Ordering::Greater; // Flexible goes first
        }
    }

    // Default: endangered goes first
    Ordering::Less
}

/// Main comparison function for sorting candidates.
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
