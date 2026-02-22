//! Core scheduling engine with candidate update and scheduling loop.

use crate::schedule::Schedule;
use crate::scheduling_block::Task;
use crate::solution_space::{Interval, SolutionSpace};
use qtty::Unit;

use super::candidate::Candidate;
use super::metrics::{compute_deadline, compute_est, compute_flexibility};

/// Updates candidate metrics and sorts them.
pub fn update_candidates<T, U>(
    candidates: &mut [Candidate<T, U>],
    solution_space: &SolutionSpace<U>,
    horizon: Interval<U>,
    endangered_threshold: u32,
) where
    T: Task<U>,
    U: Unit,
{
    // Update metrics for all candidates
    for candidate in candidates.iter_mut() {
        candidate.flexibility =
            compute_flexibility(&candidate.task, &candidate.task_id, solution_space, horizon);
        candidate.est = compute_est(&candidate.task, &candidate.task_id, solution_space, horizon);
        candidate.deadline =
            compute_deadline(&candidate.task, &candidate.task_id, solution_space, horizon);
    }

    // Sort candidates
    // Sort candidates using a total, deterministic key to avoid panics from
    // comparator inconsistencies when floating-point values (NaN) are present.
    fn f64_to_ordered_i128(x: f64) -> i128 {
        let u = x.to_bits() as i128;
        // Map IEEE-754 bit pattern to lexicographically ordered integer
        if (u as u128) >> 127 & 1 == 1 {
            !u
        } else {
            u ^ (1i128 << 63)
        }
    }

    candidates.sort_by_key(|c| {
        // impossible last
        let impossible_flag: u8 = if c.is_impossible() { 1 } else { 0 };
        // kind: endangered (0), flexible (1), other (2)
        let kind: u8 = if c.is_endangered(endangered_threshold) {
            0
        } else if c.is_flexible(endangered_threshold) {
            1
        } else {
            2
        };
        // EST key (total order). Missing EST → large value to push later.
        let est_key: i128 = c
            .est()
            .map(|q| f64_to_ordered_i128(q.value()))
            .unwrap_or(i128::MAX / 4);
        // priority: higher first → negate to sort ascending
        let prio_key: i32 = -c.task().priority();
        // flexibility key (total order)
        let flex_key: i128 = f64_to_ordered_i128(c.flexibility().value());
        // final tie-breaker: task id
        let tid = c.task_id().to_string();
        (impossible_flag, kind, est_key, prio_key, flex_key, tid)
    });
}

/// Checks if scheduling is done (no more schedulable tasks or cursor past horizon).
pub fn is_done<T, U>(
    candidates: &[Candidate<T, U>],
    cursor: qtty::Quantity<U>,
    horizon: Interval<U>,
) -> bool
where
    T: Task<U>,
    U: Unit,
{
    candidates.is_empty()
        || candidates[0].is_impossible()
        || cursor.value() >= horizon.end().value()
}

/// Finds the index of the next endangered task in the candidate list.
///
/// Returns the index of the first endangered candidate, or candidates.len() if none found.
/// Used by tests to validate endangered/flexible classification boundaries.
#[cfg(test)]
pub(crate) fn find_next_endangered_index<T, U>(
    candidates: &[Candidate<T, U>],
    endangered_threshold: u32,
) -> usize
where
    T: Task<U>,
    U: Unit,
{
    candidates
        .iter()
        .position(|c| c.is_endangered(endangered_threshold))
        .unwrap_or(candidates.len())
}

/// Schedules a segment of the horizon.
///
/// This is the main scheduling loop that repeatedly:
/// 1. Removes the highest-priority candidate
/// 2. Schedules it at its earliest start time
/// 3. Advances the cursor past the scheduled task + its gap
/// 4. Recomputes remaining candidates against the current frontier
///
/// Candidate metrics are recomputed on `[cursor, horizon.end]` at each iteration.
/// This keeps EST/deadline/flexibility aligned with the already scheduled prefix,
/// so candidates are not dropped due to stale EST values that overlap.
pub fn schedule_segment<T, U>(
    schedule: &mut Schedule<U>,
    mut candidates: Vec<Candidate<T, U>>,
    solution_space: &SolutionSpace<U>,
    horizon: Interval<U>,
    endangered_threshold: u32,
) where
    T: Task<U>,
    U: Unit,
{
    // Initialize cursor at horizon start
    let mut cursor = horizon.start();

    while !candidates.is_empty() {
        let remaining_horizon = Interval::new(cursor, horizon.end());

        // Recompute all remaining candidates against the current frontier.
        update_candidates(
            &mut candidates,
            solution_space,
            remaining_horizon,
            endangered_threshold,
        );

        if is_done(&candidates, cursor, horizon) {
            break;
        }

        let candidate = candidates.remove(0);

        // Schedule the task
        if let Some(interval) = candidate.get_interval() {
            if schedule.add(candidate.task_id(), interval).is_ok() {
                // Advance cursor to the end of the scheduled task plus any
                // required gap. Because intervals are half-open [start, end),
                // the next task may begin exactly at `interval.end()` without
                // overlapping — no epsilon offset is needed.
                cursor = interval.end() + candidate.task().gap_after();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{iv, q, TestTask};
    use qtty::Second;

    fn make_candidate(id: &str, size: f64) -> Candidate<TestTask, Second> {
        Candidate::new(TestTask::new(id, size), id)
    }

    fn make_space_for(ids: &[(&str, Vec<Interval<Second>>)]) -> SolutionSpace<Second> {
        let mut ss = SolutionSpace::new();
        for (id, intervals) in ids {
            ss.set_intervals(id.to_string(), intervals.clone());
        }
        ss
    }

    // ── update_candidates ─────────────────────────────────────────────

    #[test]
    fn update_candidates_recomputes_metrics() {
        let mut candidates = vec![make_candidate("a", 10.0), make_candidate("b", 10.0)];
        let ss = make_space_for(&[("a", vec![iv(0.0, 100.0)]), ("b", vec![iv(50.0, 100.0)])]);
        let horizon = iv(0.0, 100.0);

        update_candidates(&mut candidates, &ss, horizon, 5);

        // After update, candidates should have EST set and be sorted
        assert!(candidates[0].est().is_some());
        assert!(candidates[1].est().is_some());
    }

    #[test]
    fn update_candidates_sorts_by_priority() {
        let mut candidates = vec![make_candidate("low", 10.0), make_candidate("high", 10.0)];
        // Manually set different metrics to test sorting
        candidates[0].est = Some(q(0.0));
        candidates[0].flexibility = q(10.0);
        candidates[1].est = Some(q(0.0));
        candidates[1].flexibility = q(10.0);

        let ss = make_space_for(&[
            ("low", vec![iv(0.0, 100.0)]),
            ("high", vec![iv(0.0, 100.0)]),
        ]);

        update_candidates(&mut candidates, &ss, iv(0.0, 100.0), 5);
        // Both have same EST/priority/flexibility, sorted by ID
        assert!(candidates[0].task_id() < candidates[1].task_id());
    }

    // ── is_done ───────────────────────────────────────────────────────

    #[test]
    fn is_done_empty_candidates() {
        let candidates: Vec<Candidate<TestTask, Second>> = vec![];
        assert!(is_done(&candidates, q(0.0), iv(0.0, 100.0)));
    }

    #[test]
    fn is_done_all_impossible() {
        let candidates = vec![make_candidate("a", 10.0)];
        // est is None by default → impossible
        assert!(is_done(&candidates, q(0.0), iv(0.0, 100.0)));
    }

    #[test]
    fn is_done_cursor_past_horizon() {
        let mut candidates = vec![make_candidate("a", 10.0)];
        candidates[0].est = Some(q(0.0));
        assert!(is_done(&candidates, q(100.0), iv(0.0, 100.0)));
    }

    #[test]
    fn is_done_not_done_yet() {
        let mut candidates = vec![make_candidate("a", 10.0)];
        candidates[0].est = Some(q(0.0));
        candidates[0].flexibility = q(5.0);
        assert!(!is_done(&candidates, q(0.0), iv(0.0, 100.0)));
    }

    // ── find_next_endangered_index ────────────────────────────────────

    #[test]
    fn find_next_endangered_all_flexible() {
        let mut candidates = vec![make_candidate("a", 10.0), make_candidate("b", 10.0)];
        candidates[0].est = Some(q(0.0));
        candidates[0].flexibility = q(10.0);
        candidates[1].est = Some(q(5.0));
        candidates[1].flexibility = q(10.0);

        let index = find_next_endangered_index(&candidates, 5);
        assert_eq!(index, 2); // None found → returns len()
    }

    #[test]
    fn find_next_endangered_first_is_endangered() {
        let mut candidates = vec![make_candidate("a", 10.0)];
        candidates[0].est = Some(q(0.0));
        candidates[0].flexibility = q(2.0); // < 5

        let index = find_next_endangered_index(&candidates, 5);
        assert_eq!(index, 0);
    }

    // ── schedule_segment ──────────────────────────────────────────────

    #[test]
    fn schedule_segment_single_task() {
        let mut schedule = Schedule::new();
        let candidates = vec![make_candidate("a", 10.0)];
        let ss = make_space_for(&[("a", vec![iv(0.0, 100.0)])]);

        schedule_segment(&mut schedule, candidates, &ss, iv(0.0, 100.0), 5);

        assert_eq!(schedule.len(), 1);
        assert!(schedule.contains_task("a"));
        let interval = schedule.get_interval("a").unwrap();
        assert_eq!(interval.start().value(), 0.0);
        assert_eq!(interval.end().value(), 10.0);
    }

    #[test]
    fn schedule_segment_two_tasks_sequential() {
        let mut schedule = Schedule::new();
        let candidates = vec![make_candidate("a", 10.0), make_candidate("b", 10.0)];
        let ss = make_space_for(&[("a", vec![iv(0.0, 100.0)]), ("b", vec![iv(0.0, 100.0)])]);

        schedule_segment(&mut schedule, candidates, &ss, iv(0.0, 100.0), 5);

        assert_eq!(schedule.len(), 2);
        assert!(schedule.contains_task("a"));
        assert!(schedule.contains_task("b"));
    }

    #[test]
    fn schedule_segment_impossible_task_skipped() {
        let mut schedule = Schedule::new();
        let candidates = vec![make_candidate("impossible", 200.0)]; // too big for any window
        let ss = make_space_for(&[("impossible", vec![iv(0.0, 50.0)])]);

        schedule_segment(&mut schedule, candidates, &ss, iv(0.0, 50.0), 5);

        assert_eq!(schedule.len(), 0);
    }
}
