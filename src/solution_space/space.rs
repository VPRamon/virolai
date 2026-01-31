//! Solution space: a collection of valid intervals for task placement.
//!
//! The [`SolutionSpace`] acts as a lookup table that schedulers query to find
//! feasible positions. Users populate it with intervals computed from constraints.

use std::collections::HashMap;
use std::fmt::Display;

use super::interval::Interval;
use crate::constraints::Constraint;
use crate::Id;
use qtty::{Quantity, Unit};

/// Collection of valid intervals where tasks may be scheduled.
///
/// Maps task IDs to their valid scheduling intervals.
/// Schedulers query it to determine feasible task placements.
///
/// # Design
///
/// - Uses task IDs (`String`) as stable keys, avoiding lifetime issues
/// - Tasks without constraints get a single interval spanning [start, end]
/// - Each task maintains its own sorted, non-overlapping interval list
#[derive(Debug)]
pub struct SolutionSpace<U: Unit>(HashMap<Id, Vec<Interval<U>>>);

/// Binary search to find interval containing a position in sorted list.
fn find_interval_containing_sorted<U: Unit>(
    intervals: &[Interval<U>],
    position: Quantity<U>,
) -> Option<&Interval<U>> {
    let idx = intervals.partition_point(|i| i.end().value() < position.value());
    intervals.get(idx).filter(|i| i.contains(position))
}

impl<U: Unit> SolutionSpace<U> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Populates a solution space from multiple scheduling blocks.
    ///
    /// For each task in each block:
    /// - If the task has constraints, computes valid intervals within the given range
    /// - If the task has no constraints, uses the full range as a single interval
    ///
    /// The solution space maps task IDs to their intervals,
    /// allowing cross-block scheduling with stable task identification.
    ///
    /// # Arguments
    ///
    /// * `blocks` - One or more scheduling blocks to process
    /// * `range` - The scheduling window (interval) to compute valid placements within
    ///
    /// # Returns
    ///
    /// A [`SolutionSpace`] with intervals for all tasks in all blocks
    ///
    /// # Example
    ///
    /// ```ignore
    /// use v_rolai::solution_space::{SolutionSpace, Interval};
    /// use v_rolai::scheduling_block::SchedulingBlock;
    /// use qtty::{Quantity, Second};
    ///
    /// let block = SchedulingBlock::new();
    /// // ... add tasks ...
    ///
    /// let range = Interval::new(
    ///     Quantity::<Second>::new(0.0),
    ///     Quantity::<Second>::new(86400.0)
    /// );
    /// let solution_space = SolutionSpace::populate(&[block], range);
    /// ```
    pub fn populate<T, D, E>(
        blocks: &[crate::scheduling_block::SchedulingBlock<T, U, D, E>],
        range: Interval<U>,
    ) -> Self
    where
        T: crate::scheduling_block::Task<U>,
        E: petgraph::EdgeType,
    {
        let map = blocks
            .iter()
            .flat_map(|block| {
                block
                    .graph()
                    .node_indices()
                    .filter_map(|i| block.get_task(i))
            })
            .map(|task| {
                // Use size_on_axis() to get duration in axis units
                let task_size = task.size_on_axis();
                let intervals = task.constraints().map_or_else(
                    || vec![range],
                    |ct| {
                        ct.compute_intervals(range)
                            .into_iter()
                            .filter(|i| i.duration().value() >= task_size.value())
                            .collect()
                    },
                );
                (task.id().to_owned(), intervals)
            })
            .collect::<HashMap<Id, Vec<Interval<U>>>>();

        Self(map)
    }

    /// Adds an interval for a specific task ID.
    pub fn add_interval(&mut self, task_id: impl Into<Id>, interval: Interval<U>) {
        self.0.entry(task_id.into()).or_default().push(interval);
    }

    /// Adds multiple intervals for a specific task ID (automatically normalized).
    pub fn add_intervals(&mut self, task_id: impl Into<Id>, intervals: Vec<Interval<U>>) {
        self.0.entry(task_id.into()).or_default().extend(intervals);
    }

    /// Sets the intervals for a specific task ID, replacing any existing intervals (automatically normalized).
    pub fn set_intervals(&mut self, task_id: impl Into<Id>, intervals: Vec<Interval<U>>) {
        self.0.insert(task_id.into(), intervals);
    }

    /// Returns intervals for a specific task ID.
    pub fn get_intervals(&self, task_id: &str) -> Option<&[Interval<U>]> {
        self.0.get(task_id).map(|v| v.as_slice())
    }

    /// Returns task IDs that have intervals defined.
    pub fn task_ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.0.keys().map(|k| k.as_str())
    }

    /// Returns total number of tasks in the solution space.
    pub fn task_count(&self) -> usize {
        self.0.len()
    }

    /// Returns total number of intervals across all tasks.
    pub fn interval_count(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }

    /// Removes all intervals for a specific task ID.
    pub fn remove_task(&mut self, task_id: &str) -> bool {
        self.0.remove(task_id).is_some()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns true if the specified task ID has any interval containing `position` (O(log m) binary search).
    pub fn task_contains_position(&self, task_id: &str, position: Quantity<U>) -> bool {
        self.0
            .get(task_id)
            .map(|intervals| find_interval_containing_sorted(intervals, position).is_some())
            .unwrap_or(false)
    }

    /// Returns true if any task has an interval containing `position`.
    pub fn contains_position(&self, position: Quantity<U>) -> bool {
        self.0
            .values()
            .any(|intervals| intervals.iter().any(|interval| interval.contains(position)))
    }

    /// Returns true if the specified task ID can fit at `position` with given `size` (O(log m) binary search).
    pub fn can_place_task(&self, task_id: &str, position: Quantity<U>, size: Quantity<U>) -> bool {
        self.0
            .get(task_id)
            .map(|intervals| {
                find_interval_containing_sorted(intervals, position)
                    .map(|i| i.can_fit(position, size))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Returns true if any task can be placed at `position` with given `size`.
    pub fn can_place_at(&self, position: Quantity<U>, size: Quantity<U>) -> bool {
        self.0.values().any(|intervals| {
            intervals
                .iter()
                .any(|interval| interval.can_fit(position, size))
        })
    }

    /// Returns sum of all interval durations for a specific task ID.
    pub fn task_capacity(&self, task_id: &str) -> Quantity<U> {
        self.0
            .get(task_id)
            .map(|intervals| {
                intervals
                    .iter()
                    .map(|interval| interval.duration())
                    .fold(Quantity::new(0.0), |acc, dur| acc + dur)
            })
            .unwrap_or(Quantity::new(0.0))
    }

    /// Returns sum of all interval durations across all tasks.
    pub fn total_capacity(&self) -> Quantity<U> {
        self.0
            .values()
            .flat_map(|intervals| intervals.iter())
            .map(|interval| interval.duration())
            .fold(Quantity::new(0.0), |acc, dur| acc + dur)
    }

    /// Returns start of the first interval with capacity ≥ `size` for a specific task ID.
    pub fn find_earliest_fit_for_task(
        &self,
        task_id: &str,
        size: Quantity<U>,
    ) -> Option<Quantity<U>> {
        self.0.get(task_id).and_then(|intervals| {
            intervals
                .iter()
                .find(|interval| interval.duration().value() >= size.value())
                .map(|interval| interval.start())
        })
    }

    /// Returns start of the first interval with capacity ≥ `size` across all tasks.
    pub fn find_earliest_fit(&self, size: Quantity<U>) -> Option<Quantity<U>> {
        self.0
            .values()
            .flat_map(|intervals| intervals.iter())
            .filter(|interval| interval.duration().value() >= size.value())
            .map(|interval| interval.start())
            .min_by(|a, b| a.value().partial_cmp(&b.value()).unwrap())
    }

    /// Returns the first interval containing `position` for a specific task ID (O(log m) binary search).
    pub fn find_interval_containing_for_task(
        &self,
        task_id: &str,
        position: Quantity<U>,
    ) -> Option<&Interval<U>> {
        self.0
            .get(task_id)
            .and_then(|intervals| find_interval_containing_sorted(intervals, position))
    }

    /// Returns the first interval containing `position` across all tasks (O(log m) per task).
    pub fn find_interval_containing(&self, position: Quantity<U>) -> Option<&Interval<U>> {
        self.0
            .values()
            .find_map(|intervals| find_interval_containing_sorted(intervals, position))
    }
}

impl<U: Unit> Default for SolutionSpace<U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<U: Unit> Display for SolutionSpace<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SolutionSpace {{")?;
        writeln!(f, "  Tasks: {}", self.task_count())?;
        writeln!(f, "  Total intervals: {}", self.interval_count())?;
        writeln!(f, "  Total capacity: {:.3}", self.total_capacity().value())?;

        if !self.0.is_empty() {
            writeln!(f, "  Per-task breakdown:")?;

            for (task_id, intervals) in &self.0 {
                let capacity: Quantity<U> = intervals
                    .iter()
                    .map(|i| i.duration())
                    .fold(Quantity::new(0.0), |acc, dur| acc + dur);
                writeln!(
                    f,
                    "    task_id {}: {} intervals, capacity {:.3}",
                    task_id,
                    intervals.len(),
                    capacity.value()
                )?;
                for (i, interval) in intervals.iter().enumerate() {
                    writeln!(f, "      [{}] {}", i, interval)?;
                }
            }
        }

        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ConstraintExpr, IntervalConstraint};
    use crate::scheduling_block::{SchedulingBlock, Task};
    use qtty::{Quantity, Second};

    #[derive(Debug)]
    struct TestTask {
        id: String,
        name: String,
        size: Quantity<Second>,
        constraints: Option<ConstraintExpr<IntervalConstraint<Second>>>,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn size(&self) -> Quantity<Second> {
            self.size
        }

        fn constraints(&self) -> Option<&ConstraintExpr<IntervalConstraint<Second>>> {
            self.constraints.as_ref()
        }
    }

    #[test]
    fn test_new_solution_space() {
        let space: SolutionSpace<Second> = SolutionSpace::new();
        assert!(space.is_empty());
        assert_eq!(space.task_count(), 0);
        assert_eq!(space.interval_count(), 0);
    }

    #[test]
    fn test_populate_empty_blocks() {
        let blocks: Vec<SchedulingBlock<TestTask, Second>> = vec![];
        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let solution_space = SolutionSpace::populate(&blocks, range);
        assert_eq!(solution_space.task_count(), 0);
        assert_eq!(solution_space.interval_count(), 0);
    }

    #[test]
    fn test_populate_no_constraints() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: None,
        };
        let idx = block.add_task(task);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.task_count(), 1);
        assert_eq!(solution_space.interval_count(), 1);

        let task_ref = blocks[0].get_task(idx).unwrap();
        let intervals = solution_space.get_intervals(task_ref.id()).unwrap();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 100.0);
    }

    #[test]
    fn test_populate_with_constraints() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let constraint = IntervalConstraint::new(Interval::from_f64(10.0, 50.0));
        let task = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint)),
        };
        let idx = block.add_task(task);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.task_count(), 1);
        assert_eq!(solution_space.interval_count(), 1);

        let task_ref = blocks[0].get_task(idx).unwrap();
        let intervals = solution_space.get_intervals(task_ref.id()).unwrap();
        assert_eq!(intervals[0].start().value(), 10.0);
        assert_eq!(intervals[0].end().value(), 50.0);
    }

    #[test]
    fn test_populate_multiple_tasks() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let constraint1 = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 50.0));
        let task1 = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint1)),
        };
        let idx1 = block.add_task(task1);

        let constraint2 = IntervalConstraint::new(Interval::<Second>::from_f64(60.0, 100.0));
        let task2 = TestTask {
            id: "2".to_string(),
            name: "Task2".to_string(),
            size: Quantity::<Second>::new(15.0),
            constraints: Some(ConstraintExpr::leaf(constraint2)),
        };
        let idx2 = block.add_task(task2);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(200.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.task_count(), 2);
        assert_eq!(solution_space.interval_count(), 2);

        let task1_ref = blocks[0].get_task(idx1).unwrap();
        let task2_ref = blocks[0].get_task(idx2).unwrap();
        assert!(solution_space.get_intervals(task1_ref.id()).is_some());
        assert!(solution_space.get_intervals(task2_ref.id()).is_some());
    }

    #[test]
    fn test_task_queries() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let task1 = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: None,
        };
        let idx1 = block.add_task(task1);

        let task2 = TestTask {
            id: "2".to_string(),
            name: "Task2".to_string(),
            size: Quantity::<Second>::new(15.0),
            constraints: None,
        };
        let idx2 = block.add_task(task2);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        let task1_ref = blocks[0].get_task(idx1).unwrap();
        let task2_ref = blocks[0].get_task(idx2).unwrap();

        // Test task_contains_position
        assert!(
            solution_space.task_contains_position(task1_ref.id(), Quantity::<Second>::new(50.0))
        );
        assert!(
            solution_space.task_contains_position(task2_ref.id(), Quantity::<Second>::new(50.0))
        );

        // Test can_place_task
        assert!(solution_space.can_place_task(
            task1_ref.id(),
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(50.0)
        ));
        assert!(!solution_space.can_place_task(
            task1_ref.id(),
            Quantity::<Second>::new(60.0),
            Quantity::<Second>::new(50.0)
        ));

        // Test task_capacity
        assert_eq!(solution_space.task_capacity(task1_ref.id()).value(), 100.0);
        assert_eq!(solution_space.task_capacity(task2_ref.id()).value(), 100.0);

        // Test total_capacity
        assert_eq!(solution_space.total_capacity().value(), 200.0);
    }

    #[test]
    fn test_contains_position() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: None,
        };
        block.add_task(task);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert!(solution_space.contains_position(Quantity::<Second>::new(50.0)));
        assert!(!solution_space.contains_position(Quantity::<Second>::new(150.0)));
    }

    #[test]
    fn test_find_earliest_fit() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let constraint1 = IntervalConstraint::new(Interval::<Second>::from_f64(5000.0, 10000.0));
        let task1 = TestTask {
            id: "1".to_string(),
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint1)),
        };
        let idx1 = block.add_task(task1);

        let constraint2 = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 1000.0));
        let task2 = TestTask {
            id: "2".to_string(),
            name: "Task2".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint2)),
        };
        block.add_task(task2);

        let range = Interval::new(
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(15000.0),
        );

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        // Test find_earliest_fit_for_task
        let task1_ref = blocks[0].get_task(idx1).unwrap();
        let fit = solution_space
            .find_earliest_fit_for_task(task1_ref.id(), Quantity::<Second>::new(500.0));
        assert!(fit.is_some());
        assert_eq!(fit.unwrap().value(), 5000.0);

        // Test find_earliest_fit across all tasks
        let fit_all = solution_space.find_earliest_fit(Quantity::<Second>::new(500.0));
        assert!(fit_all.is_some());
        assert_eq!(fit_all.unwrap().value(), 0.0); // task2's interval starts earlier
    }
}
