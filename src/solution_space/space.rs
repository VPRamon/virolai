//! Solution space: a collection of valid intervals for task placement.
//!
//! The [`SolutionSpace`] acts as a lookup table that schedulers query to find
//! feasible positions. Users populate it with intervals computed from constraints.

use std::collections::HashMap;
use std::fmt::Display;

use super::interval::Interval;
use super::interval_set::IntervalSet;
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
pub struct SolutionSpace<U: Unit>(HashMap<Id, IntervalSet<U>>);

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

    /// Creates a [`SolutionSpace`] from a pre-built map.
    ///
    /// Each vector is **normalized** (sorted by start, overlapping intervals
    /// merged) so that all binary-search queries remain correct.
    pub fn from_hashmap(map: HashMap<Id, Vec<Interval<U>>>) -> Self {
        let canonical = map
            .into_iter()
            .map(|(id, intervals)| (id, IntervalSet::from(intervals)))
            .collect();
        Self(canonical)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Adds an interval for a specific ID.
    ///
    /// The stored set is kept canonical (sorted, overlaps merged) after
    /// insertion so that binary-search queries remain correct.
    pub fn add_interval(&mut self, id: impl Into<Id>, interval: Interval<U>) {
        self.0.entry(id.into()).or_default().push(interval);
    }

    /// Adds multiple intervals for a specific ID.
    ///
    /// The stored set is kept canonical (sorted, overlaps merged) after
    /// insertion so that binary-search queries remain correct.
    pub fn add_intervals(&mut self, id: impl Into<Id>, intervals: Vec<Interval<U>>) {
        self.0.entry(id.into()).or_default().extend(intervals);
    }

    /// Sets the intervals for a specific ID, replacing any existing intervals.
    ///
    /// The supplied list is normalized (sorted, overlaps merged) before storage.
    pub fn set_intervals(&mut self, id: impl Into<Id>, intervals: Vec<Interval<U>>) {
        self.0.insert(id.into(), IntervalSet::from(intervals));
    }

    /// Returns intervals for a specific ID.
    pub fn get_intervals(&self, id: &str) -> Option<&IntervalSet<U>> {
        self.0.get(id)
    }

    /// Returns IDs that have intervals defined.
    pub fn ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.0.keys().map(|k| k.as_str())
    }

    /// Returns total number of entries in the solution space.
    pub fn count(&self) -> usize {
        self.0.len()
    }

    /// Returns total number of intervals across all tasks.
    pub fn interval_count(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }

    /// Removes all intervals for a specific ID.
    pub fn remove(&mut self, id: &str) -> bool {
        self.0.remove(id).is_some()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns true if the specified ID has any interval containing `position` (O(log m) binary search).
    pub fn contains_position_for(&self, id: &str, position: Quantity<U>) -> bool {
        self.0
            .get(id)
            .map(|set| find_interval_containing_sorted(set, position).is_some())
            .unwrap_or(false)
    }

    /// Returns true if any task has an interval containing `position`.
    pub fn contains_position(&self, position: Quantity<U>) -> bool {
        self.0
            .values()
            .any(|intervals| intervals.iter().any(|interval| interval.contains(position)))
    }

    /// Returns true if the specified ID can fit at `position` with given `size` (O(log m) binary search).
    pub fn can_place(&self, id: &str, position: Quantity<U>, size: Quantity<U>) -> bool {
        self.0
            .get(id)
            .map(|set| {
                find_interval_containing_sorted(set, position)
                    .map(|i| i.can_fit(position, size))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Returns true if any entry can be placed at `position` with given `size`.
    pub fn can_place_at(&self, position: Quantity<U>, size: Quantity<U>) -> bool {
        self.0.values().any(|intervals| {
            intervals
                .iter()
                .any(|interval| interval.can_fit(position, size))
        })
    }

    /// Returns sum of all interval durations for a specific ID.
    pub fn capacity(&self, id: &str) -> Quantity<U> {
        self.0
            .get(id)
            .map(|intervals| {
                intervals
                    .iter()
                    .map(|interval| interval.duration())
                    .fold(Quantity::new(0.0), |acc, dur| acc + dur)
            })
            .unwrap_or(Quantity::new(0.0))
    }

    /// Returns sum of all interval durations across all entries.
    pub fn total_capacity(&self) -> Quantity<U> {
        self.0
            .values()
            .flat_map(|intervals| intervals.iter())
            .map(|interval| interval.duration())
            .fold(Quantity::new(0.0), |acc, dur| acc + dur)
    }

    /// Returns start of the first interval with capacity ≥ `size` for a specific ID.
    pub fn find_earliest_fit_for(&self, id: &str, size: Quantity<U>) -> Option<Quantity<U>> {
        self.0.get(id).and_then(|intervals| {
            intervals
                .iter()
                .find(|interval| interval.duration().value() >= size.value())
                .map(|interval| interval.start())
        })
    }

    /// Returns start of the first interval with capacity ≥ `size` across all entries.
    pub fn find_earliest_fit(&self, size: Quantity<U>) -> Option<Quantity<U>> {
        self.0
            .values()
            .flat_map(|intervals| intervals.iter())
            .filter(|interval| interval.duration().value() >= size.value())
            .map(|interval| interval.start())
            .min_by(|a, b| a.value().partial_cmp(&b.value()).unwrap())
    }

    /// Returns the first interval containing `position` for a specific ID (O(log m) binary search).
    pub fn find_interval_containing_for(
        &self,
        id: &str,
        position: Quantity<U>,
    ) -> Option<&Interval<U>> {
        self.0
            .get(id)
            .and_then(|set| find_interval_containing_sorted(set, position))
    }

    /// Returns the first interval containing `position` across all entries (O(log m) per entry).
    pub fn find_interval_containing(&self, position: Quantity<U>) -> Option<&Interval<U>> {
        self.0
            .values()
            .find_map(|set| find_interval_containing_sorted(set, position))
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
        writeln!(f, "  Entries: {}", self.count())?;
        writeln!(f, "  Total intervals: {}", self.interval_count())?;
        writeln!(f, "  Total capacity: {:.3}", self.total_capacity().value())?;

        if !self.0.is_empty() {
            writeln!(f, "  Per-entry breakdown:")?;

            for (id, set) in &self.0 {
                let capacity: Quantity<U> = set
                    .iter()
                    .map(|i| i.duration())
                    .fold(Quantity::new(0.0), |acc, dur| acc + dur);
                writeln!(
                    f,
                    "    id {}: {} intervals, capacity {:.3}",
                    id,
                    set.len(),
                    capacity.value()
                )?;
                for (i, interval) in set.iter().enumerate() {
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
        name: String,
        size: Quantity<Second>,
        constraints: Option<ConstraintExpr<IntervalConstraint<Second>>>,
    }

    impl Task<Second> for TestTask {
        type SizeUnit = Second;
        type ConstraintLeaf = IntervalConstraint<Second>;

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
        assert_eq!(space.count(), 0);
        assert_eq!(space.interval_count(), 0);
    }

    #[test]
    fn test_populate_empty_blocks() {
        let blocks: Vec<SchedulingBlock<TestTask, Second>> = vec![];
        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let solution_space = SolutionSpace::populate(&blocks, range);
        assert_eq!(solution_space.count(), 0);
        assert_eq!(solution_space.interval_count(), 0);
    }

    #[test]
    fn test_populate_no_constraints() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task = TestTask {
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: None,
        };
        let task_id = block.add_task(task);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.count(), 1);
        assert_eq!(solution_space.interval_count(), 1);

        let intervals = solution_space.get_intervals(&task_id).unwrap();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 100.0);
    }

    #[test]
    fn test_populate_with_constraints() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let constraint = IntervalConstraint::new(Interval::from_f64(10.0, 50.0));
        let task = TestTask {
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint)),
        };
        let task_id = block.add_task(task);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.count(), 1);
        assert_eq!(solution_space.interval_count(), 1);

        let intervals = solution_space.get_intervals(&task_id).unwrap();
        assert_eq!(intervals[0].start().value(), 10.0);
        assert_eq!(intervals[0].end().value(), 50.0);
    }

    #[test]
    fn test_populate_multiple_tasks() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let constraint1 = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 50.0));
        let task1 = TestTask {
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint1)),
        };
        let task1_id = block.add_task(task1);

        let constraint2 = IntervalConstraint::new(Interval::<Second>::from_f64(60.0, 100.0));
        let task2 = TestTask {
            name: "Task2".to_string(),
            size: Quantity::<Second>::new(15.0),
            constraints: Some(ConstraintExpr::leaf(constraint2)),
        };
        let task2_id = block.add_task(task2);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(200.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        assert_eq!(solution_space.count(), 2);
        assert_eq!(solution_space.interval_count(), 2);

        assert!(solution_space.get_intervals(&task1_id).is_some());
        assert!(solution_space.get_intervals(&task2_id).is_some());
    }

    #[test]
    fn test_task_queries() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();

        let task1 = TestTask {
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: None,
        };
        let task1_id = block.add_task(task1);

        let task2 = TestTask {
            name: "Task2".to_string(),
            size: Quantity::<Second>::new(15.0),
            constraints: None,
        };
        let task2_id = block.add_task(task2);

        let range = Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(100.0));

        let blocks = vec![block];
        let solution_space = SolutionSpace::populate(&blocks, range);

        // Test contains_position_for
        assert!(solution_space.contains_position_for(&task1_id, Quantity::<Second>::new(50.0)));
        assert!(solution_space.contains_position_for(&task2_id, Quantity::<Second>::new(50.0)));

        // Test can_place
        assert!(solution_space.can_place(
            &task1_id,
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(50.0)
        ));
        assert!(!solution_space.can_place(
            &task1_id,
            Quantity::<Second>::new(60.0),
            Quantity::<Second>::new(50.0)
        ));

        // Test capacity
        assert_eq!(solution_space.capacity(&task1_id).value(), 100.0);
        assert_eq!(solution_space.capacity(&task2_id).value(), 100.0);

        // Test total_capacity
        assert_eq!(solution_space.total_capacity().value(), 200.0);
    }

    #[test]
    fn test_contains_position() {
        let mut block: SchedulingBlock<TestTask, Second> = SchedulingBlock::new();
        let task = TestTask {
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
            name: "Task1".to_string(),
            size: Quantity::<Second>::new(10.0),
            constraints: Some(ConstraintExpr::leaf(constraint1)),
        };
        let task1_id = block.add_task(task1);

        let constraint2 = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 1000.0));
        let task2 = TestTask {
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

        // Test find_earliest_fit_for
        let fit = solution_space.find_earliest_fit_for(&task1_id, Quantity::<Second>::new(500.0));
        assert!(fit.is_some());
        assert_eq!(fit.unwrap().value(), 5000.0);

        // Test find_earliest_fit across all tasks
        let fit_all = solution_space.find_earliest_fit(Quantity::<Second>::new(500.0));
        assert!(fit_all.is_some());
        assert_eq!(fit_all.unwrap().value(), 0.0); // task2's interval starts earlier
    }

    // ── Gap coverage tests ────────────────────────────────────────────

    #[test]
    fn test_add_interval() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        assert_eq!(space.count(), 1);
        assert_eq!(space.interval_count(), 1);

        space.add_interval("task1", Interval::from_f64(60.0, 100.0));
        assert_eq!(space.count(), 1);
        assert_eq!(space.interval_count(), 2);
    }

    #[test]
    fn test_add_intervals() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_intervals(
            "task1",
            vec![
                Interval::from_f64(0.0, 50.0),
                Interval::from_f64(60.0, 100.0),
            ],
        );
        assert_eq!(space.count(), 1);
        assert_eq!(space.interval_count(), 2);
    }

    #[test]
    fn test_remove() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        assert!(space.remove("task1"));
        assert!(space.is_empty());
        assert!(!space.remove("task1")); // Already removed
    }

    #[test]
    fn test_clear() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        space.add_interval("task2", Interval::from_f64(0.0, 50.0));
        assert_eq!(space.count(), 2);
        space.clear();
        assert!(space.is_empty());
        assert_eq!(space.count(), 0);
    }

    #[test]
    fn test_is_empty() {
        let space: SolutionSpace<Second> = SolutionSpace::new();
        assert!(space.is_empty());
    }

    #[test]
    fn test_can_place_at() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 100.0));

        assert!(space.can_place_at(Quantity::new(0.0), Quantity::new(50.0)));
        assert!(!space.can_place_at(Quantity::new(60.0), Quantity::new(50.0)));
    }

    #[test]
    fn test_can_place_at_empty_space() {
        let space: SolutionSpace<Second> = SolutionSpace::new();
        assert!(!space.can_place_at(Quantity::new(0.0), Quantity::new(10.0)));
    }

    #[test]
    fn test_find_interval_containing_for() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_intervals(
            "task1",
            vec![
                Interval::from_f64(0.0, 50.0),
                Interval::from_f64(60.0, 100.0),
            ],
        );

        let found = space.find_interval_containing_for("task1", Quantity::new(70.0));
        assert!(found.is_some());
        assert_eq!(found.unwrap().start().value(), 60.0);

        let not_found = space.find_interval_containing_for("task1", Quantity::new(55.0));
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_interval_containing_for_unknown_id() {
        let space: SolutionSpace<Second> = SolutionSpace::new();
        assert!(space
            .find_interval_containing_for("nope", Quantity::new(0.0))
            .is_none());
    }

    #[test]
    fn test_find_interval_containing_across_entries() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        space.add_interval("task2", Interval::from_f64(60.0, 100.0));

        let found = space.find_interval_containing(Quantity::new(70.0));
        assert!(found.is_some());

        let not_found = space.find_interval_containing(Quantity::new(55.0));
        assert!(not_found.is_none());
    }

    #[test]
    fn test_default_impl() {
        let space: SolutionSpace<Second> = SolutionSpace::default();
        assert!(space.is_empty());
    }

    #[test]
    fn test_display() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        let output = format!("{}", space);
        assert!(output.contains("SolutionSpace"));
        assert!(output.contains("Entries: 1"));
        assert!(output.contains("Total intervals: 1"));
    }

    #[test]
    fn test_with_capacity() {
        let space: SolutionSpace<Second> = SolutionSpace::with_capacity(10);
        assert!(space.is_empty());
    }

    #[test]
    fn test_ids_iteration() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("alpha", Interval::from_f64(0.0, 50.0));
        space.add_interval("beta", Interval::from_f64(60.0, 100.0));
        let mut ids: Vec<_> = space.ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_capacity_multiple_intervals() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_intervals(
            "task1",
            vec![
                Interval::from_f64(0.0, 50.0),
                Interval::from_f64(60.0, 100.0),
            ],
        );
        assert_eq!(space.capacity("task1").value(), 90.0); // 50 + 40
    }

    #[test]
    fn test_capacity_unknown_id() {
        let space: SolutionSpace<Second> = SolutionSpace::new();
        assert_eq!(space.capacity("nope").value(), 0.0);
    }

    #[test]
    fn test_set_intervals_replaces() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        space.set_intervals("task1", vec![Interval::from_f64(10.0, 20.0)]);
        assert_eq!(space.interval_count(), 1);
        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals[0].start().value(), 10.0);
    }

    // ── Invariant-enforcement regression tests ────────────────────────

    /// Out-of-order insertion would previously corrupt binary search results.
    #[test]
    fn test_add_interval_out_of_order_is_sorted() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        // Insert later interval first.
        space.add_interval("task1", Interval::from_f64(60.0, 100.0));
        space.add_interval("task1", Interval::from_f64(0.0, 40.0));

        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals.len(), 2);
        assert!(
            intervals[0].start().value() < intervals[1].start().value(),
            "intervals must be sorted by start"
        );

        // Binary-search query must find the correct interval.
        assert!(space.contains_position_for("task1", Quantity::new(10.0)));
        assert!(space.contains_position_for("task1", Quantity::new(80.0)));
        assert!(!space.contains_position_for("task1", Quantity::new(50.0)));
    }

    /// Overlapping intervals must be merged so binary search stays valid.
    #[test]
    fn test_add_interval_overlapping_are_merged() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 60.0));
        space.add_interval("task1", Interval::from_f64(40.0, 100.0));

        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals.len(), 1, "overlapping intervals must be merged");
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 100.0);
    }

    /// Touching (abutting) intervals must also be merged.
    #[test]
    fn test_add_interval_touching_are_merged() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_interval("task1", Interval::from_f64(0.0, 50.0));
        space.add_interval("task1", Interval::from_f64(50.0, 100.0));

        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals.len(), 1, "touching intervals must be merged");
        assert_eq!(intervals[0].end().value(), 100.0);
    }

    /// add_intervals with an unsorted batch must still produce a sorted list.
    #[test]
    fn test_add_intervals_unsorted_batch() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.add_intervals(
            "task1",
            vec![
                Interval::from_f64(200.0, 300.0),
                Interval::from_f64(0.0, 100.0),
                Interval::from_f64(50.0, 150.0), // overlaps previous
            ],
        );

        let intervals = space.get_intervals("task1").unwrap();
        // [0,150] and [200,300] – first two merged, third separate
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 150.0);
        assert_eq!(intervals[1].start().value(), 200.0);
    }

    /// set_intervals must normalize even when given unsorted input.
    #[test]
    fn test_set_intervals_normalizes() {
        let mut space: SolutionSpace<Second> = SolutionSpace::new();
        space.set_intervals(
            "task1",
            vec![
                Interval::from_f64(80.0, 100.0),
                Interval::from_f64(0.0, 30.0),
                Interval::from_f64(20.0, 50.0), // overlaps previous
            ],
        );

        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 50.0);
        assert_eq!(intervals[1].start().value(), 80.0);

        // Binary-search query must work correctly.
        assert!(space.contains_position_for("task1", Quantity::new(25.0)));
        assert!(!space.contains_position_for("task1", Quantity::new(60.0)));
    }

    /// from_hashmap must normalize every entry it receives.
    #[test]
    fn test_from_hashmap_normalizes() {
        use std::collections::HashMap;
        let mut map: HashMap<String, Vec<Interval<Second>>> = HashMap::new();
        map.insert(
            "task1".to_string(),
            vec![
                Interval::from_f64(50.0, 100.0),
                Interval::from_f64(0.0, 60.0), // out of order AND overlapping
            ],
        );

        let space = SolutionSpace::from_hashmap(map);
        let intervals = space.get_intervals("task1").unwrap();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 100.0);
    }
}
