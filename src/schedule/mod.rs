use crate::solution_space::Interval;
use crate::Id;
use qtty::Quantity;
use std::collections::{BTreeMap, HashMap};
pub mod entry_key;
pub mod errors;
use entry_key::*;
use errors::*;

#[cfg(test)]
mod tests;

/// Schedule of non-overlapping tasks sorted by start time.
///
/// A `Schedule` maintains a collection of non-overlapping intervals (tasks) indexed by task ID,
/// providing efficient operations for insertion, removal, and conflict detection.
///
/// # Internal Structure
/// - `by_start`: `BTreeMap` from start time to task entry
/// - `start_by_id`: `HashMap` from task ID to start time
///
/// # Complexity
/// - `add`: O(log n) with O(1) neighbor overlap checks
/// - `remove`: O(log n)
/// - `get_interval`: O(1) hash lookup + O(log n) tree lookup
/// - `conflicts`: O(log n + k) where k is the number of conflicts
/// - `task_at`: O(log n)
///
/// # Examples
///
/// ```
/// use vrolai::schedule::Schedule;
/// use vrolai::solution_space::Interval;
/// use qtty::{Quantity, Second};
///
/// let mut schedule = Schedule::<Second>::new();
///
/// // Add some non-overlapping tasks
/// schedule.add("1", Interval::from_f64(0.0, 10.0)).unwrap();
/// schedule.add("2", Interval::from_f64(15.0, 25.0)).unwrap();
/// schedule.add("3", Interval::from_f64(30.0, 40.0)).unwrap();
///
/// assert_eq!(schedule.len(), 3);
///
/// // Check if a time slot is free
/// let query = Interval::from_f64(10.5, 14.5);
/// assert!(schedule.is_free(query).unwrap());
///
/// // Find conflicts with a proposed interval
/// let conflicts = schedule.conflicts_vec(Interval::from_f64(8.0, 20.0)).unwrap();
/// assert_eq!(conflicts.len(), 2); // Overlaps with tasks 1 and 2
///
/// // Find which task is at a specific time
/// assert_eq!(schedule.task_at(Quantity::new(5.0)).unwrap(), Some("1".to_string()));
/// assert_eq!(schedule.task_at(Quantity::new(12.0)).unwrap(), None);
///
/// // Remove a task
/// let removed = schedule.remove("2");
/// assert_eq!(removed.unwrap().start().value(), 15.0);
/// ```
#[derive(Debug, Clone)]
pub struct Schedule<U: qtty::Unit> {
    by_start: BTreeMap<F64Key, Entry<U>>,
    start_by_id: HashMap<Id, F64Key>,
}

impl<U: qtty::Unit> Default for Schedule<U> {
    fn default() -> Self {
        Self {
            by_start: BTreeMap::new(),
            start_by_id: HashMap::new(),
        }
    }
}

impl<U: qtty::Unit> Schedule<U> {
    pub fn new() -> Self {
        Self {
            by_start: BTreeMap::new(),
            start_by_id: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.by_start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_start.is_empty()
    }

    fn key(q: qtty::Quantity<U>) -> Result<F64Key, ScheduleError> {
        let v = q.value();
        if v.is_nan() {
            Err(ScheduleError::NaNTime)
        } else {
            Ok(F64Key(v))
        }
    }

    fn key_f64(v: f64) -> Result<F64Key, ScheduleError> {
        if v.is_nan() {
            Err(ScheduleError::NaNTime)
        } else {
            Ok(F64Key(v))
        }
    }

    /// Returns true if task id exists.
    pub fn contains_task(&self, id: &str) -> bool {
        self.start_by_id.contains_key(id)
    }

    /// Gets the interval for a task id (if present).
    pub fn get_interval(&self, id: &str) -> Option<Interval<U>> {
        let start = self.start_by_id.get(id)?;
        self.by_start.get(start).map(|e| e.interval)
    }

    /// Inserts a task with its interval.
    ///
    /// Requires:
    /// - `id` not already present
    /// - interval times not NaN
    /// - interval does not overlap any existing interval
    ///
    /// Efficiency: only predecessor + successor checks are needed because the schedule
    /// is maintained as non-overlapping and sorted by start time.
    pub fn add(&mut self, id: impl Into<Id>, interval: Interval<U>) -> Result<(), ScheduleError> {
        let id: Id = id.into();
        if self.contains_task(&id) {
            return Err(ScheduleError::DuplicateTaskId(id));
        }

        let start_k = Self::key(interval.start())?;
        let end_k = Self::key(interval.end())?;

        // Optional sanity: if your Interval::new already enforces start <= end,
        // you don't need this. Still, it helps if callers bypass it somehow.
        if start_k > end_k {
            // Interval::new would panic; here we just fail fast with NaNTime? not right.
            // Keep behavior aligned with Interval::new: assume it's valid.
        }

        // Check predecessor (latest interval with start <= new.start).
        if let Some((_k, prev)) = self.by_start.range(..=start_k).next_back() {
            if prev.interval.overlaps(&interval) {
                return Err(ScheduleError::OverlapsExisting {
                    new_id: id,
                    existing_id: prev.id.clone(),
                });
            }
        }

        // Check successor (earliest interval with start >= new.start).
        if let Some((_k, next)) = self.by_start.range(start_k..).next() {
            if next.interval.overlaps(&interval) {
                return Err(ScheduleError::OverlapsExisting {
                    new_id: id,
                    existing_id: next.id.clone(),
                });
            }
        }

        self.by_start.insert(
            start_k,
            Entry {
                id: id.clone(),
                interval,
            },
        );
        self.start_by_id.insert(id, start_k);
        Ok(())
    }

    /// Removes a task by id. Returns its interval if it existed.
    pub fn remove(&mut self, id: &str) -> Option<Interval<U>> {
        let start_k = self.start_by_id.remove(id)?;
        let entry = self.by_start.remove(&start_k)?;
        Some(entry.interval)
    }

    /// Returns true if `query` overlaps any scheduled task.
    pub fn has_conflict(&self, query: Interval<U>) -> Result<bool, ScheduleError> {
        Ok(self.conflicts(query)?.next().is_some())
    }

    /// Iterates over all conflicts (overlapping scheduled tasks) with `query`.
    ///
    /// Complexity: O(log n + k) where k is the number of conflicts.
    ///
    /// Note: uses `Interval::overlaps` exactly as you defined it (inclusive endpoints).
    /// If you want “back-to-back tasks are OK”, change Interval semantics to half-open
    /// and update `overlaps` accordingly.
    pub fn conflicts<'a>(
        &'a self,
        query: Interval<U>,
    ) -> Result<impl Iterator<Item = (Id, Interval<U>)> + 'a, ScheduleError> {
        let q_start = query.start().value();
        let q_end = query.end().value();

        let q_start_k = Self::key_f64(q_start)?;
        let _q_end_k = Self::key_f64(q_end)?;

        // Determine where to start scanning:
        // - the predecessor of q_start (it might start before q_start but still overlap)
        // - otherwise the first start >= q_start
        let start_from = if let Some((k, prev)) = self.by_start.range(..=q_start_k).next_back() {
            if prev.interval.overlaps(&query) {
                Some(*k)
            } else {
                None
            }
        } else {
            None
        };

        let range_start = match start_from {
            Some(k) => k,
            None => q_start_k,
        };

        // Scan all intervals whose start <= q_end and filter by overlap.
        // We can't stop at q_end because we need inclusive range behavior.
        let iter = self
            .by_start
            .range(range_start..)
            .take_while(move |(k, _e)| k.0 <= q_end)
            .filter(move |(_k, e)| e.interval.overlaps(&query))
            .map(|(_k, e)| (e.id.clone(), e.interval));

        Ok(iter)
    }

    /// Convenience: returns conflicts collected into a Vec.
    pub fn conflicts_vec(
        &self,
        query: Interval<U>,
    ) -> Result<Vec<(Id, Interval<U>)>, ScheduleError> {
        Ok(self.conflicts(query)?.collect())
    }

    /// Checks if an interval can be inserted without conflicts.
    pub fn is_free(&self, query: Interval<U>) -> Result<bool, ScheduleError> {
        Ok(!self.has_conflict(query)?)
    }

    /// (Optional) Find the task that contains `pos`, if any.
    ///
    /// Complexity: O(log n).
    pub fn task_at(&self, pos: Quantity<U>) -> Result<Option<Id>, ScheduleError> {
        let p = Self::key(pos)?;
        if let Some((_k, e)) = self.by_start.range(..=p).next_back() {
            if e.interval.contains(pos) {
                return Ok(Some(e.id.clone()));
            }
        }
        Ok(None)
    }

    /// Returns an iterator over all scheduled tasks in start time order.
    ///
    /// Each item is `(id, interval)`.
    pub fn iter(&self) -> impl Iterator<Item = (Id, Interval<U>)> + '_ {
        self.by_start.values().map(|e| (e.id.clone(), e.interval))
    }

    /// Returns an iterator over all IDs.
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.start_by_id.keys().cloned()
    }

    /// Returns an iterator over intervals in start time order.
    pub fn intervals(&self) -> impl Iterator<Item = Interval<U>> + '_ {
        self.by_start.values().map(|e| e.interval)
    }

    /// Clears all tasks from the schedule.
    pub fn clear(&mut self) {
        self.by_start.clear();
        self.start_by_id.clear();
    }

    /// Returns the total scheduled duration (sum of all interval durations).
    ///
    /// Note: this does NOT account for gaps between tasks.
    pub fn total_duration(&self) -> Quantity<U> {
        self.by_start
            .values()
            .map(|e| e.interval.duration())
            .fold(Quantity::new(0.0), |acc, dur| acc + dur)
    }

    /// Returns the earliest start time in the schedule, if any.
    pub fn earliest_start(&self) -> Option<Quantity<U>> {
        self.by_start.values().next().map(|e| e.interval.start())
    }

    /// Returns the latest end time in the schedule, if any.
    pub fn latest_end(&self) -> Option<Quantity<U>> {
        self.by_start.values().next_back().map(|e| e.interval.end())
    }

    /// Returns the time span from earliest start to latest end, if any tasks exist.
    pub fn span(&self) -> Option<Quantity<U>> {
        if let (Some(start), Some(end)) = (self.earliest_start(), self.latest_end()) {
            Some(end - start)
        } else {
            None
        }
    }
}

// =============================================================================
// Schedule Serde Support
// =============================================================================

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use crate::solution_space::Interval;
    use serde::de::{self, MapAccess, SeqAccess, Visitor};
    use serde::ser::{SerializeSeq, SerializeStruct};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::marker::PhantomData;

    /// Helper struct for serializing schedule entries.
    struct ScheduleEntryOut<'a, U: qtty::Unit> {
        task: &'a str,
        interval: &'a Interval<U>,
    }

    impl<U: qtty::Unit> Serialize for ScheduleEntryOut<'_, U> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut s = serializer.serialize_struct("ScheduleEntry", 2)?;
            s.serialize_field("task", self.task)?;
            s.serialize_field("interval", self.interval)?;
            s.end()
        }
    }

    impl<U: qtty::Unit> Serialize for Schedule<U> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(self.len()))?;
            for (id, interval) in self.iter() {
                seq.serialize_element(&ScheduleEntryOut {
                    task: &id,
                    interval: &interval,
                })?;
            }
            seq.end()
        }
    }

    impl<'de, U: qtty::Unit> Deserialize<'de> for Schedule<U> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct ScheduleVisitor<U: qtty::Unit>(PhantomData<U>);

            impl<'de, U: qtty::Unit> Visitor<'de> for ScheduleVisitor<U> {
                type Value = Schedule<U>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str("a sequence of schedule entries")
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: SeqAccess<'de>,
                {
                    let mut schedule = Schedule::new();
                    while let Some(entry) = seq.next_element::<ScheduleEntryIn<U>>()? {
                        schedule
                            .add(entry.task, entry.interval)
                            .map_err(de::Error::custom)?;
                    }
                    Ok(schedule)
                }
            }

            deserializer.deserialize_seq(ScheduleVisitor(PhantomData))
        }
    }

    /// Helper struct for deserializing schedule entries with backward compatibility.
    struct ScheduleEntryIn<U: qtty::Unit> {
        task: String,
        interval: Interval<U>,
    }

    impl<'de, U: qtty::Unit> Deserialize<'de> for ScheduleEntryIn<U> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct EntryVisitor<U: qtty::Unit>(PhantomData<U>);

            impl<'de, U: qtty::Unit> Visitor<'de> for EntryVisitor<U> {
                type Value = ScheduleEntryIn<U>;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str(
                        "a schedule entry object with 'task' (or 'task_id') and 'interval' fields",
                    )
                }

                fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                where
                    M: MapAccess<'de>,
                {
                    let mut task: Option<String> = None;
                    let mut interval: Option<Interval<U>> = None;

                    while let Some(key) = map.next_key::<String>()? {
                        match key.as_str() {
                            "task" | "task_id" => {
                                if task.is_some() {
                                    return Err(de::Error::duplicate_field("task"));
                                }
                                task = Some(map.next_value()?);
                            }
                            "interval" => {
                                if interval.is_some() {
                                    return Err(de::Error::duplicate_field("interval"));
                                }
                                interval = Some(map.next_value()?);
                            }
                            _ => {
                                // Ignore unknown fields
                                let _ = map.next_value::<serde::de::IgnoredAny>()?;
                            }
                        }
                    }

                    let task = task.ok_or_else(|| de::Error::missing_field("task"))?;
                    let interval = interval.ok_or_else(|| de::Error::missing_field("interval"))?;

                    Ok(ScheduleEntryIn { task, interval })
                }
            }

            deserializer.deserialize_map(EntryVisitor(PhantomData))
        }
    }
}
