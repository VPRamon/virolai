//! Task candidate with computed scheduling metrics.

use crate::scheduling_block::Task;
use crate::solution_space::Interval;
use qtty::{Quantity, Unit};

/// Candidate task for scheduling with computed metrics.
#[derive(Debug, Clone)]
pub struct Candidate<T, U>
where
    T: Task<U>,
    U: Unit,
{
    pub(crate) task: T,
    pub(crate) task_id: u64,
    pub(crate) est: Option<Quantity<U>>,
    pub(crate) deadline: Option<Quantity<U>>,
    pub(crate) flexibility: Quantity<U>,
}

impl<T, U> Candidate<T, U>
where
    T: Task<U>,
    U: Unit,
{
    /// Creates a new candidate with uninitialized metrics.
    pub fn new(task: T) -> Self {
        let task_id = task.id();
        Self {
            task,
            task_id,
            est: None,
            deadline: None,
            flexibility: Quantity::new(0.0),
        }
    }

    /// Returns true if the task cannot be scheduled (no EST found).
    pub fn is_impossible(&self) -> bool {
        self.est.is_none()
    }

    /// Returns true if the task is endangered (low flexibility).
    pub fn is_endangered(&self, threshold: u32) -> bool {
        !self.is_impossible() && self.flexibility.value() <= threshold as f64
    }

    /// Returns true if the task is flexible (high flexibility).
    pub fn is_flexible(&self, threshold: u32) -> bool {
        !self.is_impossible() && self.flexibility.value() > threshold as f64
    }

    /// Get the scheduling period for this candidate.
    pub fn get_period(&self) -> Option<Interval<U>> {
        self.est
            .map(|start| Interval::new(start, start + self.task.size()))
    }

    /// Get task reference.
    pub fn task(&self) -> &T {
        &self.task
    }

    /// Get task ID.
    pub fn task_id(&self) -> u64 {
        self.task_id
    }

    /// Get earliest start time.
    pub fn est(&self) -> Option<Quantity<U>> {
        self.est
    }

    /// Get deadline.
    pub fn deadline(&self) -> Option<Quantity<U>> {
        self.deadline
    }

    /// Get flexibility.
    pub fn flexibility(&self) -> Quantity<U> {
        self.flexibility
    }
}
