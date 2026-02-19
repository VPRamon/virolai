//! Comprehensive test suite for Schedule module.

use super::*;
use qtty::Second;

type TestInterval = Interval<Second>;
type TestSchedule = Schedule<Second>;

/// Helper to create intervals more concisely in tests.
fn iv(start: f64, end: f64) -> TestInterval {
    Interval::from_f64(start, end)
}

/// Helper to create a quantity.
fn q(value: f64) -> Quantity<Second> {
    Quantity::new(value)
}

#[cfg(test)]
mod basic_operations {
    use super::*;

    #[test]
    fn test_new_schedule_is_empty() {
        let schedule = TestSchedule::new();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);
    }

    #[test]
    fn test_add_single_task() {
        let mut schedule = TestSchedule::new();
        let result = schedule.add("1", iv(0.0, 10.0));
        assert!(result.is_ok());
        assert_eq!(schedule.len(), 1);
        assert!(schedule.contains_task("1"));
    }

    #[test]
    fn test_add_duplicate_task_id_fails() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        let result = schedule.add("1", iv(20.0, 30.0));
        assert_eq!(result, Err(ScheduleError::DuplicateTaskId("1".to_string())));
    }

    #[test]
    fn test_get_interval() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert_eq!(schedule.get_interval("1"), Some(iv(0.0, 10.0)));
        assert_eq!(schedule.get_interval("2"), Some(iv(20.0, 30.0)));
        assert_eq!(schedule.get_interval("999"), None);
    }

    #[test]
    fn test_remove_task() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        let removed = schedule.remove("1");
        assert_eq!(removed, Some(iv(0.0, 10.0)));
        assert_eq!(schedule.len(), 1);
        assert!(!schedule.contains_task("1"));
        assert!(schedule.contains_task("2"));
    }

    #[test]
    fn test_remove_nonexistent_task() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        let removed = schedule.remove("999");
        assert_eq!(removed, None);
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        schedule.clear();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);
    }
}

#[cfg(test)]
mod overlap_detection {
    use super::*;

    #[test]
    fn test_non_overlapping_tasks() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(10.1, 20.0)).unwrap();
        schedule.add("3", iv(20.1, 30.0)).unwrap();

        assert_eq!(schedule.len(), 3);
    }

    #[test]
    fn test_overlapping_tasks_rejected() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        // Overlaps at the end
        let result = schedule.add("2", iv(5.0, 15.0));
        assert!(
            matches!(result, Err(ScheduleError::OverlapsExisting { new_id, existing_id }) if new_id == "2" && existing_id == "1")
        );
    }

    #[test]
    fn test_overlapping_with_multiple_tasks() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        // Try to add task that overlaps with task 2
        let result = schedule.add("4", iv(25.0, 35.0));
        assert!(
            matches!(result, Err(ScheduleError::OverlapsExisting { new_id, existing_id }) if new_id == "4" && existing_id == "2")
        );
    }

    #[test]
    fn test_touching_intervals_at_boundary() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        // Half-open: [0, 10) and [10, 20) share no point → back-to-back is NOT an overlap.
        let result = schedule.add("2", iv(10.0, 20.0));
        assert!(
            result.is_ok(),
            "Back-to-back half-open tasks must not conflict"
        );
    }

    #[test]
    fn test_contained_interval_rejected() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 100.0)).unwrap();

        // Task completely contained within existing task
        let result = schedule.add("2", iv(10.0, 20.0));
        assert!(
            matches!(result, Err(ScheduleError::OverlapsExisting { new_id, existing_id }) if new_id == "2" && existing_id == "1")
        );
    }

    #[test]
    fn test_containing_interval_rejected() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(10.0, 20.0)).unwrap();

        // Task that contains existing task
        let result = schedule.add("2", iv(0.0, 100.0));
        assert!(
            matches!(result, Err(ScheduleError::OverlapsExisting { new_id, existing_id }) if new_id == "2" && existing_id == "1")
        );
    }
}

#[cfg(test)]
mod conflict_queries {
    use super::*;

    #[test]
    fn test_has_conflict_with_overlapping() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert!(schedule.has_conflict(iv(5.0, 15.0)).unwrap());
        assert!(schedule.has_conflict(iv(25.0, 35.0)).unwrap());
    }

    #[test]
    fn test_has_conflict_no_overlap() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert!(!schedule.has_conflict(iv(10.1, 19.9)).unwrap());
        assert!(!schedule.has_conflict(iv(30.1, 40.0)).unwrap());
    }

    #[test]
    fn test_conflicts_vec_single() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        let conflicts = schedule.conflicts_vec(iv(5.0, 15.0)).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].0, "1");
    }

    #[test]
    fn test_conflicts_vec_multiple() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        let conflicts = schedule.conflicts_vec(iv(5.0, 45.0)).unwrap();
        assert_eq!(conflicts.len(), 3);
        assert_eq!(conflicts[0].0, "1");
        assert_eq!(conflicts[1].0, "2");
        assert_eq!(conflicts[2].0, "3");
    }

    #[test]
    fn test_conflicts_vec_none() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        let conflicts = schedule.conflicts_vec(iv(10.1, 19.9)).unwrap();
        assert_eq!(conflicts.len(), 0);
    }

    #[test]
    fn test_is_free() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert!(schedule.is_free(iv(10.1, 19.9)).unwrap());
        assert!(!schedule.is_free(iv(5.0, 15.0)).unwrap());
    }
}

#[cfg(test)]
mod task_at_position {
    use super::*;

    #[test]
    fn test_task_at_finds_task() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert_eq!(schedule.task_at(q(5.0)).unwrap(), Some("1".to_string()));
        assert_eq!(schedule.task_at(q(25.0)).unwrap(), Some("2".to_string()));
    }

    #[test]
    fn test_task_at_boundaries() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        // Half-open [0, 10): start is included, end is excluded.
        assert_eq!(schedule.task_at(q(0.0)).unwrap(), Some("1".to_string()));
        assert_eq!(schedule.task_at(q(9.999)).unwrap(), Some("1".to_string()));
        assert_eq!(schedule.task_at(q(10.0)).unwrap(), None);
    }

    #[test]
    fn test_task_at_no_task() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        assert_eq!(schedule.task_at(q(15.0)).unwrap(), None);
        assert_eq!(schedule.task_at(q(35.0)).unwrap(), None);
    }
}

#[cfg(test)]
mod iterators {
    use super::*;

    #[test]
    fn test_iter_order() {
        let mut schedule = TestSchedule::new();
        schedule.add("3", iv(40.0, 50.0)).unwrap();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        let tasks: Vec<_> = schedule.iter().collect();
        assert_eq!(tasks.len(), 3);
        // Should be sorted by start time
        assert_eq!(tasks[0].0, "1");
        assert_eq!(tasks[1].0, "2");
        assert_eq!(tasks[2].0, "3");
    }

    #[test]
    fn test_task_ids() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        let mut ids: Vec<_> = schedule.ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    }

    #[test]
    fn test_intervals() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();

        let intervals: Vec<_> = schedule.intervals().collect();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0], iv(0.0, 10.0));
        assert_eq!(intervals[1], iv(20.0, 30.0));
    }
}

#[cfg(test)]
mod statistics {
    use super::*;

    #[test]
    fn test_total_duration() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        assert_eq!(schedule.total_duration().value(), 30.0);
    }

    #[test]
    fn test_total_duration_empty() {
        let schedule = TestSchedule::new();
        assert_eq!(schedule.total_duration().value(), 0.0);
    }

    #[test]
    fn test_earliest_start() {
        let mut schedule = TestSchedule::new();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        assert_eq!(schedule.earliest_start().unwrap().value(), 0.0);
    }

    #[test]
    fn test_earliest_start_empty() {
        let schedule = TestSchedule::new();
        assert_eq!(schedule.earliest_start(), None);
    }

    #[test]
    fn test_latest_end() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 50.0)).unwrap();

        assert_eq!(schedule.latest_end().unwrap().value(), 50.0);
    }

    #[test]
    fn test_latest_end_empty() {
        let schedule = TestSchedule::new();
        assert_eq!(schedule.latest_end(), None);
    }

    #[test]
    fn test_span() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.add("2", iv(20.0, 30.0)).unwrap();
        schedule.add("3", iv(40.0, 50.0)).unwrap();

        assert_eq!(schedule.span().unwrap().value(), 50.0);
    }

    #[test]
    fn test_span_empty() {
        let schedule = TestSchedule::new();
        assert_eq!(schedule.span(), None);
    }

    #[test]
    fn test_span_single_task() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(10.0, 30.0)).unwrap();

        assert_eq!(schedule.span().unwrap().value(), 20.0);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_duration_interval() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(10.0, 10.0)).unwrap();

        assert!(schedule.contains_task("1"));
        assert_eq!(schedule.get_interval("1").unwrap().duration().value(), 0.0);
    }

    #[test]
    fn test_many_tasks() {
        let mut schedule = TestSchedule::new();

        // Add 100 non-overlapping tasks
        for i in 0..100 {
            let start = i as f64 * 10.0;
            let end = start + 5.0;
            schedule.add(i.to_string(), iv(start, end)).unwrap();
        }

        assert_eq!(schedule.len(), 100);
        assert_eq!(schedule.total_duration().value(), 500.0);
    }

    #[test]
    fn test_task_ids_not_sequential() {
        let mut schedule = TestSchedule::new();
        schedule.add("1000", iv(0.0, 10.0)).unwrap();
        schedule.add("5", iv(20.0, 30.0)).unwrap();
        schedule.add("999", iv(40.0, 50.0)).unwrap();

        assert_eq!(schedule.len(), 3);
        assert!(schedule.contains_task("1000"));
        assert!(schedule.contains_task("5"));
        assert!(schedule.contains_task("999"));
    }

    #[test]
    fn test_add_remove_add_same_id() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();
        schedule.remove("1");

        // Should be able to re-add the same ID
        let result = schedule.add("1", iv(20.0, 30.0));
        assert!(result.is_ok());
        assert_eq!(schedule.get_interval("1"), Some(iv(20.0, 30.0)));
    }

    #[test]
    fn test_very_large_time_values() {
        let mut schedule = TestSchedule::new();
        let large = 1e15;
        schedule.add("1", iv(large, large + 100.0)).unwrap();
        schedule.add("2", iv(large + 200.0, large + 300.0)).unwrap();

        assert_eq!(schedule.len(), 2);
        assert!(schedule.is_free(iv(large + 101.0, large + 199.0)).unwrap());
    }

    #[test]
    fn test_negative_time_values() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(-100.0, -50.0)).unwrap();
        schedule.add("2", iv(-40.0, 0.0)).unwrap();
        schedule.add("3", iv(10.0, 20.0)).unwrap();

        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule.earliest_start().unwrap().value(), -100.0);
    }
}

#[cfg(test)]
mod nan_handling {
    use super::*;

    // Note: Interval::new panics if start > end, which happens with NaN comparisons.
    // The Schedule itself will reject NaN via the key() function if somehow an interval
    // with NaN is constructed. These tests verify that NaN values are properly rejected
    // when querying existing schedules.

    #[test]
    fn test_nan_in_query_conflicts() {
        let mut schedule = TestSchedule::new();
        schedule.add("1", iv(0.0, 10.0)).unwrap();

        // Create an interval that has NaN (this will panic in Interval::new)
        // So we test if has_conflict properly rejects NaN by checking the result
        // We can't construct such an interval through normal means, so we test
        // the key validation instead.

        // Test with task_at which accepts a single quantity
        let result = schedule.task_at(q(f64::NAN));
        assert_eq!(result, Err(ScheduleError::NaNTime));
    }

    #[test]
    fn test_infinity_values_work() {
        // Infinity is a valid f64 value and should work
        let mut schedule = TestSchedule::new();
        let result = schedule.add("1", iv(0.0, f64::INFINITY));
        assert!(result.is_ok());

        let result2 = schedule.add("2", iv(f64::NEG_INFINITY, -100.0));
        assert!(result2.is_ok());
    }
}

// =============================================================================
// Serde serialization tests
// =============================================================================

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;

    #[test]
    fn test_schedule_serialize_deserialize_roundtrip() {
        let mut schedule = TestSchedule::new();
        schedule.add("task1", iv(0.0, 10.0)).unwrap();
        schedule.add("task2", iv(20.0, 30.0)).unwrap();
        schedule.add("task3", iv(40.0, 50.0)).unwrap();

        let json = serde_json::to_string(&schedule).unwrap();
        let restored: TestSchedule = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 3);
        assert!(restored.contains_task("task1"));
        assert!(restored.contains_task("task2"));
        assert!(restored.contains_task("task3"));
        assert_eq!(restored.get_interval("task1"), Some(iv(0.0, 10.0)));
        assert_eq!(restored.get_interval("task2"), Some(iv(20.0, 30.0)));
        assert_eq!(restored.get_interval("task3"), Some(iv(40.0, 50.0)));
    }

    #[test]
    fn test_schedule_json_format() {
        let mut schedule = TestSchedule::new();
        schedule.add("obs1", iv(100.0, 200.0)).unwrap();

        let json = serde_json::to_string_pretty(&schedule).unwrap();
        assert!(json.contains("\"task\""));
        assert!(json.contains("\"obs1\""));
        assert!(json.contains("\"interval\""));
        assert!(json.contains("\"start\""));
        assert!(json.contains("\"end\""));
    }

    #[test]
    fn test_schedule_deserialize_with_task_id_alias() {
        // Test backward compatibility: "task_id" should be accepted
        let json = r#"[
            {"task_id": "old_task", "interval": {"start": 0.0, "end": 10.0}},
            {"task": "new_task", "interval": {"start": 20.0, "end": 30.0}}
        ]"#;

        let schedule: TestSchedule = serde_json::from_str(json).unwrap();

        assert_eq!(schedule.len(), 2);
        assert!(schedule.contains_task("old_task"));
        assert!(schedule.contains_task("new_task"));
        assert_eq!(schedule.get_interval("old_task"), Some(iv(0.0, 10.0)));
        assert_eq!(schedule.get_interval("new_task"), Some(iv(20.0, 30.0)));
    }

    #[test]
    fn test_schedule_deserialize_rejects_duplicates() {
        let json = r#"[
            {"task": "dup", "interval": {"start": 0.0, "end": 10.0}},
            {"task": "dup", "interval": {"start": 20.0, "end": 30.0}}
        ]"#;

        let result: Result<TestSchedule, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn test_schedule_deserialize_rejects_overlaps() {
        let json = r#"[
            {"task": "task1", "interval": {"start": 0.0, "end": 15.0}},
            {"task": "task2", "interval": {"start": 10.0, "end": 20.0}}
        ]"#;

        let result: Result<TestSchedule, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("overlaps"));
    }

    #[test]
    fn test_empty_schedule_serialize_deserialize() {
        let schedule = TestSchedule::new();
        let json = serde_json::to_string(&schedule).unwrap();
        assert_eq!(json, "[]");

        let restored: TestSchedule = serde_json::from_str(&json).unwrap();
        assert!(restored.is_empty());
    }
}
