//! Core scheduling engine with candidate update and scheduling loop.

use crate::schedule::Schedule;
use crate::scheduling_block::Task;
use crate::solution_space::{Interval, SolutionSpace};
use qtty::Unit;

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
        candidate.flexibility = compute_flexibility(&candidate.task, solution_space, horizon);
        candidate.est = compute_est(&candidate.task, solution_space, horizon);
        candidate.deadline = compute_deadline(&candidate.task, solution_space, horizon);
    }

    // Sort candidates
    candidates.sort_by(|a, b| compare_candidates(a, b, endangered_threshold));
}

/// Checks if scheduling is done (no more schedulable tasks or cursor past horizon).
pub fn is_done<T, U>(candidates: &[Candidate<T, U>], cursor: qtty::Quantity<U>, horizon: Interval<U>) -> bool
where
    T: Task<U>,
    U: Unit,
{
    candidates.is_empty() || candidates[0].is_impossible() || cursor.value() >= horizon.end().value()
}

/// Finds the index of the next endangered task in the candidate list.
///
/// Returns the index of the first endangered candidate, or candidates.len() if none found.
/// This is used for the partial update optimization - only candidates before the next
/// endangered task need to be re-evaluated after scheduling a task.
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
/// 4. Partially re-updates remaining candidates (optimization)
///
/// Unlike earlier versions, the horizon remains static throughout scheduling,
/// matching the C++ core implementation. A separate cursor tracks scheduling
/// progress and is advanced by task.end() + task.delay_after().
///
/// ## Partial Update Optimization
///
/// After scheduling a task, only candidates up to (but not including) the next
/// endangered task are re-evaluated. This assumes endangered tasks maintain
/// stable priority, so flexible tasks beyond them don't need immediate updates.
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

    while !is_done(&candidates, cursor, horizon) {
        let candidate = candidates.remove(0);

        // Schedule the task
        if let Some(period) = candidate.get_period() {
            if schedule.add(candidate.task_id(), period).is_ok() {
                // Advance cursor by task end time + delay after
                cursor = period.end() + candidate.task().delay_after();
                
                // Partial update optimization: only update up to next endangered task
                let next_endangered_idx = find_next_endangered_index(&candidates, endangered_threshold);
                if next_endangered_idx > 0 {
                    // Update candidates before the next endangered task
                    update_candidates(
                        &mut candidates[..next_endangered_idx],
                        solution_space,
                        horizon,
                        endangered_threshold,
                    );
                }
            }
        }
    }
}
