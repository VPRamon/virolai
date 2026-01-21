use crate::solution_space::Interval;

/// A total-order key for `f64` using IEEE-754 total order (`total_cmp`).
/// This lets us use `f64`-backed times as `BTreeMap` keys.
///
/// For scheduling, NaN is nonsense, so we reject NaNs on insert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F64Key(pub(crate) f64);

impl F64Key {
    /// Creates a new F64Key. Use this when you need the raw value.
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl Eq for F64Key {}

impl Ord for F64Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for F64Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// An entry in the schedule, mapping a task ID to its interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry<U: qtty::Unit> {
    pub(crate) id: u64,
    pub(crate) interval: Interval<U>,
}

impl<U: qtty::Unit> Entry<U> {
    /// Creates a new schedule entry.
    pub fn new(id: u64, interval: Interval<U>) -> Self {
        Self { id, interval }
    }

    /// Returns the task ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the interval.
    pub fn interval(&self) -> Interval<U> {
        self.interval
    }
}
