//! Metric computation functions for task scheduling.

use crate::scheduling_block::Task;
use crate::solution_space::{Interval, SolutionSpace};
use qtty::{Quantity, Unit};

/// Finds the earliest start time for a task in the solution space.
///
/// Searches through visibility windows that intersect with the horizon,
/// returning the start of the first window where the task fits.
///
/// Note: Uses `task.size_on_axis()` to get the duration in axis units.
pub fn compute_est<T, A>(
    task: &T,
    solution_space: &SolutionSpace<A>,
    horizon: Interval<A>,
) -> Option<Quantity<A>>
where
    T: Task<A>,
    A: Unit,
{
    let intervals = solution_space.get_intervals(task.id())?;
    let task_size = task.size_on_axis();

    for interval in intervals {
        // Skip windows that end before horizon begins
        if interval.end().value() <= horizon.start().value() {
            continue;
        }

        // Skip windows that start after horizon ends (intervals are sorted)
        if interval.start().value() >= horizon.end().value() {
            break;
        }

        // Compute intersection: window ∩ horizon
        if let Some(intersection) = interval.intersection(&horizon) {
            // Check if task fits within the effective window
            if intersection.duration().value() >= task_size.value() {
                return Some(intersection.start());
            }
        }
    }

    None
}

/// Finds the latest possible start time (deadline) for a task.
///
/// Searches backwards through visibility windows that intersect with the horizon,
/// returning the latest time the task can start and still fit.
///
/// Note: Uses `task.size_on_axis()` to get the duration in axis units.
pub fn compute_deadline<T, A>(
    task: &T,
    solution_space: &SolutionSpace<A>,
    horizon: Interval<A>,
) -> Option<Quantity<A>>
where
    T: Task<A>,
    A: Unit,
{
    let intervals = solution_space.get_intervals(task.id())?;
    let task_size = task.size_on_axis();

    // Search backwards from the end
    for interval in intervals.iter().rev() {
        // Skip windows that begin after horizon ends
        if interval.start().value() >= horizon.end().value() {
            continue;
        }

        // Skip windows that end before horizon begins
        if interval.end().value() <= horizon.start().value() {
            break;
        }

        // Compute intersection: window ∩ horizon
        if let Some(intersection) = interval.intersection(&horizon) {
            // Check if task fits within the effective window
            if intersection.duration().value() >= task_size.value() {
                return Some(intersection.end() - task_size);
            }
        }
    }

    None
}

/// Computes task flexibility as the ratio of available time to task duration.
///
/// Sums flexibility across all visibility windows that intersect with the horizon.
/// Assumes non-overlapping visibility windows.
///
/// Note: Uses `task.size_on_axis()` to get the duration in axis units.
pub fn compute_flexibility<T, A>(
    task: &T,
    solution_space: &SolutionSpace<A>,
    horizon: Interval<A>,
) -> Quantity<A>
where
    T: Task<A>,
    A: Unit,
{
    let intervals = solution_space.get_intervals(task.id());
    if intervals.is_none() {
        return Quantity::new(0.0);
    }

    let intervals = intervals.unwrap();
    let task_size = task.size_on_axis();
    let mut flexibility = 0.0;

    for interval in intervals {
        // Skip windows that end before horizon begins
        if interval.end().value() <= horizon.start().value() {
            continue;
        }

        // Skip windows that start after horizon ends (intervals are sorted)
        if interval.start().value() >= horizon.end().value() {
            break;
        }

        // Compute intersection: window ∩ horizon
        if let Some(intersection) = interval.intersection(&horizon) {
            let intersection_duration = intersection.duration().value();
            let task_duration = task_size.value();

            // Check if task fits within the effective window
            if task_duration <= intersection_duration {
                flexibility += intersection_duration / task_duration;
            }
        }
    }

    Quantity::new(flexibility)
}
