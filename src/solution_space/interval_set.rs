//! A canonical container for non-overlapping, sorted intervals.
//!
//! [`IntervalSet`] wraps a `Vec<Interval<U>>` and guarantees the **canonical
//! invariant** at all times: intervals are sorted by start and no two intervals
//! overlap or abut (touching intervals are merged).
//!
//! Read access is fully transparent via `Deref<Target = [Interval<U>]>`, so
//! existing code that consumes `&[Interval<U>]` works without changes.
//! Mutable access goes through dedicated methods that re-establish the invariant.

use std::fmt::Display;
use std::ops::{Deref, Index, RangeFull};

use super::interval::Interval;
use qtty::Unit;

/// A sorted, non-overlapping set of half-open intervals.
///
/// The container maintains the **canonical invariant**: intervals are sorted by
/// start, non-overlapping, and abutting intervals are merged. This is enforced
/// on construction and on every mutation.
///
/// # Transparent read access
///
/// `IntervalSet<U>` implements `Deref<Target = [Interval<U>]>`, so all
/// immutable slice methods (`.len()`, `.iter()`, indexing, `.first()`,
/// `.last()`, `.windows()`, etc.) are available directly.
///
/// # Performance
///
/// - Construction from unsorted input: O(n log n) sort + O(n) merge.
/// - `push`: O(n) worst-case (shift + local merge), O(1) amortized when
///   appending in order.
/// - `extend`: O((n+m) log(n+m)) worst-case, using a full re-normalize.
/// - Read access: O(1) via `Deref`.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSet<U: Unit>(Vec<Interval<U>>);

// ─────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> IntervalSet<U> {
    /// Creates an empty interval set.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Creates an empty interval set with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Wraps a `Vec` that is **already in canonical form** without re-sorting.
    ///
    /// In debug builds this asserts the invariant; in release builds the check
    /// is elided for zero overhead.
    ///
    /// # Safety (logical)
    ///
    /// The caller **must** ensure the input is sorted by start with no
    /// overlapping or abutting intervals. Violating this in release mode
    /// will silently produce incorrect results from downstream operations.
    pub fn from_sorted_unchecked(vec: Vec<Interval<U>>) -> Self {
        debug_assert!(
            crate::constraints::operations::assertions::is_canonical(&vec),
            "IntervalSet::from_sorted_unchecked called with non-canonical input"
        );
        Self(vec)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> IntervalSet<U> {
    /// Sorts by start and merges overlapping / touching intervals in place.
    fn normalize(&mut self) {
        if self.0.len() <= 1 {
            return;
        }
        self.0.sort_by(|a, b| {
            a.start()
                .value()
                .partial_cmp(&b.start().value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut merged: Vec<Interval<U>> = Vec::with_capacity(self.0.len());
        for interval in self.0.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.end().value() >= interval.start().value() {
                    // Overlapping or touching – extend the current run.
                    if interval.end().value() > last.end().value() {
                        *last = Interval::new(last.start(), interval.end());
                    }
                } else {
                    merged.push(interval);
                }
            } else {
                merged.push(interval);
            }
        }
        self.0 = merged;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Mutation methods
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> IntervalSet<U> {
    /// Inserts an interval, maintaining canonical form.
    ///
    /// Performs a binary-search insert followed by a local merge with
    /// neighbours. O(n) worst-case due to the shift, but O(1) amortized
    /// when intervals are appended in order.
    pub fn push(&mut self, interval: Interval<U>) {
        if self.0.is_empty() {
            self.0.push(interval);
            return;
        }

        // Fast path: append when the new interval starts at or after the last end.
        if let Some(last) = self.0.last() {
            if interval.start().value() >= last.end().value() {
                // Abutting or strictly after – possibly merge with last.
                if interval.start().value() == last.end().value() {
                    // Abutting: merge into last.
                    let last_idx = self.0.len() - 1;
                    self.0[last_idx] = Interval::new(self.0[last_idx].start(), interval.end());
                } else {
                    self.0.push(interval);
                }
                return;
            }
        }

        // General case: binary-search insert + full normalize.
        self.0.push(interval);
        self.normalize();
    }

    /// Appends all intervals from a slice, then re-normalizes.
    pub fn extend_from_slice(&mut self, intervals: &[Interval<U>]) {
        if intervals.is_empty() {
            return;
        }
        self.0.extend_from_slice(intervals);
        self.normalize();
    }

    /// Removes all intervals.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Retains only the intervals for which the predicate returns `true`.
    ///
    /// Because removal cannot violate sorted-ness or create new overlaps,
    /// the canonical invariant is preserved without re-normalization.
    pub fn retain<F: FnMut(&Interval<U>) -> bool>(&mut self, f: F) {
        self.0.retain(f);
    }

    /// Consumes the set and returns the underlying `Vec`.
    pub fn into_inner(self) -> Vec<Interval<U>> {
        self.0
    }

    /// Returns a slice of the intervals.
    pub fn as_slice(&self) -> &[Interval<U>] {
        &self.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Set operations
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> IntervalSet<U> {
    /// Returns the union of `self` and `other`.
    pub fn union(&self, other: &IntervalSet<U>) -> IntervalSet<U> {
        crate::constraints::operations::compute_union(&self.0, &other.0)
    }

    /// Returns the intersection of `self` and `other`.
    pub fn intersection(&self, other: &IntervalSet<U>) -> IntervalSet<U> {
        crate::constraints::operations::compute_intersection(&self.0, &other.0)
    }

    /// Returns the complement of `self` within `bounds`.
    pub fn complement(&self, bounds: Interval<U>) -> IntervalSet<U> {
        crate::constraints::operations::compute_complement(self.0.clone(), bounds)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Transparent read access
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> Deref for IntervalSet<U> {
    type Target = [Interval<U>];

    fn deref(&self) -> &[Interval<U>] {
        &self.0
    }
}

impl<U: Unit> AsRef<[Interval<U>]> for IntervalSet<U> {
    fn as_ref(&self) -> &[Interval<U>] {
        &self.0
    }
}

impl<U: Unit> std::borrow::Borrow<[Interval<U>]> for IntervalSet<U> {
    fn borrow(&self) -> &[Interval<U>] {
        &self.0
    }
}

impl<U: Unit> Index<usize> for IntervalSet<U> {
    type Output = Interval<U>;

    fn index(&self, index: usize) -> &Interval<U> {
        &self.0[index]
    }
}

impl<U: Unit> Index<RangeFull> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, _: RangeFull) -> &[Interval<U>] {
        &self.0
    }
}

impl<U: Unit> Index<std::ops::Range<usize>> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, range: std::ops::Range<usize>) -> &[Interval<U>] {
        &self.0[range]
    }
}

impl<U: Unit> Index<std::ops::RangeFrom<usize>> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, range: std::ops::RangeFrom<usize>) -> &[Interval<U>] {
        &self.0[range]
    }
}

impl<U: Unit> Index<std::ops::RangeTo<usize>> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, range: std::ops::RangeTo<usize>) -> &[Interval<U>] {
        &self.0[range]
    }
}

impl<U: Unit> Index<std::ops::RangeInclusive<usize>> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, range: std::ops::RangeInclusive<usize>) -> &[Interval<U>] {
        &self.0[range]
    }
}

impl<U: Unit> Index<std::ops::RangeToInclusive<usize>> for IntervalSet<U> {
    type Output = [Interval<U>];

    fn index(&self, range: std::ops::RangeToInclusive<usize>) -> &[Interval<U>] {
        &self.0[range]
    }
}

// ─────────────────────────────────────────────────────────────────────
// Conversions
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> From<Vec<Interval<U>>> for IntervalSet<U> {
    /// Creates an `IntervalSet` from an unsorted `Vec`, normalizing on construction.
    fn from(mut vec: Vec<Interval<U>>) -> Self {
        let mut set = Self(Vec::new());
        std::mem::swap(&mut set.0, &mut vec);
        set.normalize();
        set
    }
}

impl<U: Unit> From<Interval<U>> for IntervalSet<U> {
    /// Creates a single-element `IntervalSet` (always canonical).
    fn from(interval: Interval<U>) -> Self {
        Self(vec![interval])
    }
}

impl<U: Unit> FromIterator<Interval<U>> for IntervalSet<U> {
    fn from_iter<I: IntoIterator<Item = Interval<U>>>(iter: I) -> Self {
        let vec: Vec<Interval<U>> = iter.into_iter().collect();
        Self::from(vec)
    }
}

impl<U: Unit> Extend<Interval<U>> for IntervalSet<U> {
    fn extend<I: IntoIterator<Item = Interval<U>>>(&mut self, iter: I) {
        self.0.extend(iter);
        self.normalize();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Iterators
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> IntoIterator for IntervalSet<U> {
    type Item = Interval<U>;
    type IntoIter = std::vec::IntoIter<Interval<U>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, U: Unit> IntoIterator for &'a IntervalSet<U> {
    type Item = &'a Interval<U>;
    type IntoIter = std::slice::Iter<'a, Interval<U>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────

impl<U: Unit> Default for IntervalSet<U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<U: Unit> Display for IntervalSet<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, interval) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", interval)?;
        }
        write!(f, "}}")
    }
}

/// Enables `assert_eq!(interval_set, vec![...])` in tests.
impl<U: Unit> PartialEq<Vec<Interval<U>>> for IntervalSet<U> {
    fn eq(&self, other: &Vec<Interval<U>>) -> bool {
        self.0 == *other
    }
}

/// Enables `assert_eq!(vec![...], interval_set)` in tests.
impl<U: Unit> PartialEq<IntervalSet<U>> for Vec<Interval<U>> {
    fn eq(&self, other: &IntervalSet<U>) -> bool {
        *self == other.0
    }
}

// ─────────────────────────────────────────────────────────────────────
// Serde support
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "serde")]
impl<U: Unit> serde::Serialize for IntervalSet<U> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, U: Unit> serde::Deserialize<'de> for IntervalSet<U> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec = Vec::<Interval<U>>::deserialize(deserializer)?;
        Ok(Self::from(vec))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::Second;

    fn iv(start: f64, end: f64) -> Interval<Second> {
        Interval::from_f64(start, end)
    }

    // ── Construction ──────────────────────────────────────────────────

    #[test]
    fn new_is_empty() {
        let set = IntervalSet::<Second>::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn with_capacity_is_empty() {
        let set = IntervalSet::<Second>::with_capacity(10);
        assert!(set.is_empty());
    }

    #[test]
    fn from_empty_vec() {
        let set = IntervalSet::<Second>::from(Vec::new());
        assert!(set.is_empty());
    }

    #[test]
    fn from_single_interval() {
        let set = IntervalSet::from(iv(0.0, 10.0));
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 10.0));
    }

    #[test]
    fn from_sorted_non_overlapping() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 10.0));
        assert_eq!(set[1], iv(20.0, 30.0));
    }

    #[test]
    fn from_unsorted_normalizes() {
        let set = IntervalSet::from(vec![iv(20.0, 30.0), iv(0.0, 10.0)]);
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 10.0));
        assert_eq!(set[1], iv(20.0, 30.0));
    }

    #[test]
    fn from_overlapping_merges() {
        let set = IntervalSet::from(vec![iv(0.0, 60.0), iv(40.0, 100.0)]);
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 100.0));
    }

    #[test]
    fn from_abutting_merges() {
        let set = IntervalSet::from(vec![iv(0.0, 50.0), iv(50.0, 100.0)]);
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 100.0));
    }

    #[test]
    fn from_multiple_overlapping_and_unsorted() {
        let set = IntervalSet::from(vec![iv(200.0, 300.0), iv(0.0, 100.0), iv(50.0, 150.0)]);
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 150.0));
        assert_eq!(set[1], iv(200.0, 300.0));
    }

    #[test]
    fn from_sorted_unchecked_canonical() {
        let set = IntervalSet::from_sorted_unchecked(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn from_iterator() {
        let set: IntervalSet<Second> = vec![iv(20.0, 30.0), iv(0.0, 10.0)].into_iter().collect();
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 10.0));
    }

    // ── Push ──────────────────────────────────────────────────────────

    #[test]
    fn push_into_empty() {
        let mut set = IntervalSet::<Second>::new();
        set.push(iv(10.0, 20.0));
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(10.0, 20.0));
    }

    #[test]
    fn push_appends_in_order() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0)]);
        set.push(iv(20.0, 30.0));
        assert_eq!(set.len(), 2);
        assert_eq!(set[1], iv(20.0, 30.0));
    }

    #[test]
    fn push_abutting_merges() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0)]);
        set.push(iv(10.0, 20.0));
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 20.0));
    }

    #[test]
    fn push_overlapping_merges() {
        let mut set = IntervalSet::from(vec![iv(0.0, 50.0)]);
        set.push(iv(30.0, 80.0));
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 80.0));
    }

    #[test]
    fn push_before_existing_normalizes() {
        let mut set = IntervalSet::from(vec![iv(50.0, 100.0)]);
        set.push(iv(0.0, 30.0));
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 30.0));
        assert_eq!(set[1], iv(50.0, 100.0));
    }

    #[test]
    fn push_bridging_multiple_merges() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        set.push(iv(5.0, 25.0));
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 30.0));
    }

    // ── Extend ────────────────────────────────────────────────────────

    #[test]
    fn extend_from_slice_normalizes() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0)]);
        set.extend_from_slice(&[iv(30.0, 40.0), iv(5.0, 15.0)]);
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(0.0, 15.0));
        assert_eq!(set[1], iv(30.0, 40.0));
    }

    #[test]
    fn extend_trait_normalizes() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0)]);
        set.extend(vec![iv(5.0, 20.0)]);
        assert_eq!(set.len(), 1);
        assert_eq!(set[0], iv(0.0, 20.0));
    }

    // ── Clear / Retain ────────────────────────────────────────────────

    #[test]
    fn clear_empties() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    fn retain_filters_preserving_order() {
        let mut set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0), iv(40.0, 50.0)]);
        set.retain(|i| i.start().value() >= 15.0);
        assert_eq!(set.len(), 2);
        assert_eq!(set[0], iv(20.0, 30.0));
        assert_eq!(set[1], iv(40.0, 50.0));
    }

    // ── into_inner ────────────────────────────────────────────────────

    #[test]
    fn into_inner_returns_vec() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        let vec = set.into_inner();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], iv(0.0, 10.0));
    }

    // ── Deref / AsRef / Borrow ────────────────────────────────────────

    #[test]
    fn deref_provides_slice_methods() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert_eq!(set.first().unwrap(), &iv(0.0, 10.0));
        assert_eq!(set.last().unwrap(), &iv(20.0, 30.0));
        let _iter_count = set.iter().count();
        assert_eq!(_iter_count, 2);
    }

    #[test]
    fn coerces_to_slice_ref() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0)]);
        fn accepts_slice<U: Unit>(_s: &[Interval<U>]) {}
        accepts_slice(&set);
    }

    // ── Set operations ────────────────────────────────────────────────

    #[test]
    fn union_disjoint() {
        let a = IntervalSet::from(vec![iv(0.0, 10.0)]);
        let b = IntervalSet::from(vec![iv(20.0, 30.0)]);
        let u = a.union(&b);
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn union_overlapping() {
        let a = IntervalSet::from(vec![iv(0.0, 50.0)]);
        let b = IntervalSet::from(vec![iv(30.0, 80.0)]);
        let u = a.union(&b);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0], iv(0.0, 80.0));
    }

    #[test]
    fn intersection_overlapping() {
        let a = IntervalSet::from(vec![iv(0.0, 50.0)]);
        let b = IntervalSet::from(vec![iv(30.0, 80.0)]);
        let i = a.intersection(&b);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0], iv(30.0, 50.0));
    }

    #[test]
    fn intersection_disjoint() {
        let a = IntervalSet::from(vec![iv(0.0, 10.0)]);
        let b = IntervalSet::from(vec![iv(20.0, 30.0)]);
        let i = a.intersection(&b);
        assert!(i.is_empty());
    }

    #[test]
    fn complement_full() {
        let set = IntervalSet::from(vec![iv(20.0, 40.0), iv(60.0, 80.0)]);
        let c = set.complement(iv(0.0, 100.0));
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], iv(0.0, 20.0));
        assert_eq!(c[1], iv(40.0, 60.0));
        assert_eq!(c[2], iv(80.0, 100.0));
    }

    // ── Display ───────────────────────────────────────────────────────

    #[test]
    fn display_format() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        let s = format!("{}", set);
        assert!(s.starts_with('{'));
        assert!(s.ends_with('}'));
    }

    // ── Default ───────────────────────────────────────────────────────

    #[test]
    fn default_is_empty() {
        let set = IntervalSet::<Second>::default();
        assert!(set.is_empty());
    }

    // ── PartialEq<Vec> ───────────────────────────────────────────────

    #[test]
    fn partial_eq_with_vec() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        let vec = vec![iv(0.0, 10.0), iv(20.0, 30.0)];
        assert_eq!(set, vec);
        assert_eq!(vec, set);
    }

    // ── IntoIterator ──────────────────────────────────────────────────

    #[test]
    fn into_iter_owned() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        let collected: Vec<_> = set.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn into_iter_borrowed() {
        let set = IntervalSet::from(vec![iv(0.0, 10.0), iv(20.0, 30.0)]);
        let collected: Vec<_> = (&set).into_iter().collect();
        assert_eq!(collected.len(), 2);
    }
}
