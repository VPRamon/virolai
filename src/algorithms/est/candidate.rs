//! Task candidate with computed scheduling metrics.

use crate::scheduling_block::Task;
use crate::solution_space::Interval;
use crate::Id;
use qtty::{Quantity, Unit};

/// Candidate task for scheduling with computed metrics.
#[derive(Debug, Clone)]
pub struct Candidate<T, A>
where
    T: Task<A>,
    A: Unit,
{
    pub(crate) task: T,
    pub(crate) task_id: Id,
    pub(crate) est: Option<Quantity<A>>,
    pub(crate) deadline: Option<Quantity<A>>,
    pub(crate) flexibility: Quantity<A>,
}

impl<T, A> Candidate<T, A>
where
    T: Task<A>,
    A: Unit,
{
    /// Creates a new candidate with uninitialized metrics.
    pub fn new(task: T) -> Self {
        let task_id = task.id().to_owned();
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

    /// Get the scheduling period for this candidate (in axis units).
    pub fn get_period(&self) -> Option<Interval<A>> {
        self.est
            .map(|start| Interval::new(start, start + self.task.size_on_axis()))
    }

    /// Get task reference.
    pub fn task(&self) -> &T {
        &self.task
    }

    /// Get task ID.
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Get earliest start time.
    pub fn est(&self) -> Option<Quantity<A>> {
        self.est
    }

    /// Get deadline.
    pub fn deadline(&self) -> Option<Quantity<A>> {
        self.deadline
    }

    /// Get flexibility.
    pub fn flexibility(&self) -> Quantity<A> {
        self.flexibility
    }
}
