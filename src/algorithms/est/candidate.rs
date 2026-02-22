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
    /// Creates a new candidate with the given task and ID.
    pub fn new(task: T, task_id: impl Into<Id>) -> Self {
        Self {
            task,
            task_id: task_id.into(),
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
    /// Uses strict less-than to match C++ core implementation.
    pub fn is_endangered(&self, threshold: u32) -> bool {
        !self.is_impossible() && self.flexibility.value() < threshold as f64
    }

    /// Returns true if the task is flexible (high flexibility).
    /// Uses greater-than-or-equal to match C++ core implementation.
    pub fn is_flexible(&self, threshold: u32) -> bool {
        !self.is_impossible() && self.flexibility.value() >= threshold as f64
    }

    /// Get the scheduled interval for this candidate (in axis units).
    pub fn get_interval(&self) -> Option<Interval<A>> {
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
    #[allow(dead_code)]
    pub fn deadline(&self) -> Option<Quantity<A>> {
        self.deadline
    }

    /// Get flexibility.
    pub fn flexibility(&self) -> Quantity<A> {
        self.flexibility
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTask;
    use qtty::Second;

    #[test]
    fn new_candidate_defaults() {
        let c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "task-1");
        assert_eq!(c.task_id(), "task-1");
        assert_eq!(c.task().name(), "t");
        assert!(c.est().is_none());
        assert!(c.deadline().is_none());
        assert_eq!(c.flexibility().value(), 0.0);
    }

    #[test]
    fn is_impossible_when_no_est() {
        let c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        assert!(c.is_impossible());
    }

    #[test]
    fn not_impossible_with_est() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(0.0));
        assert!(!c.is_impossible());
    }

    #[test]
    fn is_endangered_low_flexibility() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(0.0));
        c.flexibility = Quantity::new(3.0);
        assert!(c.is_endangered(5)); // 3 < 5
        assert!(!c.is_flexible(5));
    }

    #[test]
    fn is_flexible_high_flexibility() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(0.0));
        c.flexibility = Quantity::new(10.0);
        assert!(c.is_flexible(5)); // 10 >= 5
        assert!(!c.is_endangered(5));
    }

    #[test]
    fn impossible_is_neither_endangered_nor_flexible() {
        let c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        // est is None → impossible
        assert!(!c.is_endangered(5));
        assert!(!c.is_flexible(5));
    }

    #[test]
    fn get_interval_returns_correct_interval() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(5.0));
        let interval = c.get_interval().unwrap();
        assert_eq!(interval.start().value(), 5.0);
        assert_eq!(interval.end().value(), 15.0); // 5 + 10
    }

    #[test]
    fn get_interval_none_when_impossible() {
        let c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        assert!(c.get_interval().is_none());
    }

    #[test]
    fn est_and_deadline_accessors() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(5.0));
        c.deadline = Some(Quantity::new(90.0));
        assert_eq!(c.est().unwrap().value(), 5.0);
        assert_eq!(c.deadline().unwrap().value(), 90.0);
    }

    #[test]
    fn flexibility_at_threshold_boundary() {
        let mut c = Candidate::<TestTask, Second>::new(TestTask::new("t", 10.0), "t");
        c.est = Some(Quantity::new(0.0));
        c.flexibility = Quantity::new(5.0);
        // Exactly at threshold: >= 5 is flexible, < 5 is endangered
        assert!(c.is_flexible(5));
        assert!(!c.is_endangered(5));
    }
}
