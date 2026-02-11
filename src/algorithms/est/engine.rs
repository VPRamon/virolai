//! Core scheduling engine with candidate update and scheduling loop.

use crate::schedule::Schedule;
use crate::scheduling_block::Task;
use crate::solution_space::{Interval, SolutionSpace};
use qtty::{Quantity, Unit};

use super::candidate::Candidate;
use super::metrics::{compute_deadline, compute_est, compute_flexibility};
use super::ordering::compare_candidates;

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
        candidate.flexibility = compute_flexibility(&candidate.task, &candidate.task_id, solution_space, horizon);
        candidate.est = compute_est(&candidate.task, &candidate.task_id, solution_space, horizon);
        candidate.deadline = compute_deadline(&candidate.task, &candidate.task_id, solution_space, horizon);
    }

    // Sort candidates
    candidates.sort_by(|a, b| compare_candidates(a, b, endangered_threshold));
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
/// 3. Advances the cursor past the scheduled task + its delay
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
    const CURSOR_EPSILON: f64 = 1e-6;

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
        if let Some(period) = candidate.get_period() {
            if schedule.add(candidate.task_id(), period).is_ok() {
                // Advance cursor by task end time + delay after
                //
                // Intervals are inclusive ([start, end]), so starting a new task at exactly
                // the previous end is treated as overlap. A tiny epsilon enforces strict
                // forward progress for the next frontier.
                cursor = period.end()
                    + candidate.task().delay_after()
                    + Quantity::<U>::new(CURSOR_EPSILON);
            }
        }
    }
}
