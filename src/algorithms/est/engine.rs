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

/// Checks if scheduling is done (no more schedulable tasks).
pub fn is_done<T, U>(candidates: &[Candidate<T, U>]) -> bool
where
    T: Task<U>,
    U: Unit,
{
    candidates.is_empty() || candidates[0].is_impossible()
}

/// Schedules a segment of the horizon.
///
/// This is the main scheduling loop that repeatedly:
/// 1. Removes the highest-priority candidate
/// 2. Schedules it at its earliest start time
/// 3. Moves the horizon forward past the scheduled task
/// 4. Re-updates remaining candidates with the new horizon
pub fn schedule_segment<T, U>(
    schedule: &mut Schedule<U>,
    mut candidates: Vec<Candidate<T, U>>,
    solution_space: &SolutionSpace<U>,
    mut horizon: Interval<U>,
    endangered_threshold: u32,
) where
    T: Task<U>,
    U: Unit,
{
    while !is_done(&candidates) {
        let candidate = candidates.remove(0);

        // Schedule the task
        if let Some(period) = candidate.get_period() {
            if schedule.add(candidate.task_id(), period).is_ok() {
                // Move horizon to start after the scheduled task
                let new_start = period.end();
                if new_start.value() < horizon.end().value() {
                    horizon = Interval::new(new_start, horizon.end());
                    // Update remaining candidates with new horizon
                    update_candidates(
                        &mut candidates,
                        solution_space,
                        horizon,
                        endangered_threshold,
                    );
                } else {
                    break; // No more time in horizon
                }
            }
        }
    }
}
