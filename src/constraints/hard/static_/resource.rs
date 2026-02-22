//! Resource constraint — restricts a task to eligible resources by ID and/or type.
//!
//! This is a **hard + static** constraint: it determines at pre-scheduling time
//! which resources can host a given task. Because resource eligibility is not
//! time-dependent, `compute_intervals` returns the full scheduling range —
//! the actual filtering is performed by the prescheduler, which inspects these
//! constraints to decide which (resource, task) pairs to populate.

use std::collections::HashSet;
use std::fmt;

use crate::solution_space::{Interval, IntervalSet};
use qtty::Unit;

use super::constraint::Constraint;

/// Constrains a task to run only on resources whose ID or type matches.
///
/// - `allowed_ids`: if `Some`, the task may only be scheduled on resources whose
///   `resource_id()` is in the set.
/// - `allowed_types`: if `Some`, the task may only be scheduled on resources whose
///   `resource_type()` is in the set.
///
/// When both are `Some`, a resource is eligible if it matches **either** (union).
/// When both are `None`, the constraint is vacuous (all resources are eligible).
///
/// # Example
///
/// ```
/// use virolai::constraints::ResourceConstraint;
///
/// // Task that can only run on LST1 or LST2
/// let by_id = ResourceConstraint::from_ids(["LST1", "LST2"]);
/// assert!(by_id.matches("LST1", "LST"));
/// assert!(!by_id.matches("MAGIC1", "MAGIC"));
///
/// // Task that can run on any telescope of type "LST"
/// let by_type = ResourceConstraint::from_types(["LST"]);
/// assert!(by_type.matches("LST3", "LST"));
/// assert!(!by_type.matches("MAGIC1", "MAGIC"));
/// ```
#[derive(Debug, Clone)]
pub struct ResourceConstraint {
    /// Allowed resource identifiers (e.g., `"LST1"`, `"MAGIC2"`).
    allowed_ids: Option<HashSet<String>>,
    /// Allowed resource type/category labels (e.g., `"LST"`, `"MAGIC"`).
    allowed_types: Option<HashSet<String>>,
}

impl ResourceConstraint {
    /// Creates a resource constraint with explicit ID and type sets.
    pub fn new(
        allowed_ids: Option<HashSet<String>>,
        allowed_types: Option<HashSet<String>>,
    ) -> Self {
        Self {
            allowed_ids,
            allowed_types,
        }
    }

    /// Creates a resource constraint allowing only the given resource IDs.
    pub fn from_ids(ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_ids: Some(ids.into_iter().map(Into::into).collect()),
            allowed_types: None,
        }
    }

    /// Creates a resource constraint allowing only the given resource types.
    pub fn from_types(types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_ids: None,
            allowed_types: Some(types.into_iter().map(Into::into).collect()),
        }
    }

    /// Creates a resource constraint allowing either specific IDs or types (union).
    pub fn from_ids_and_types(
        ids: impl IntoIterator<Item = impl Into<String>>,
        types: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            allowed_ids: Some(ids.into_iter().map(Into::into).collect()),
            allowed_types: Some(types.into_iter().map(Into::into).collect()),
        }
    }

    /// Returns the allowed resource IDs, if specified.
    pub fn allowed_ids(&self) -> Option<&HashSet<String>> {
        self.allowed_ids.as_ref()
    }

    /// Returns the allowed resource types, if specified.
    pub fn allowed_types(&self) -> Option<&HashSet<String>> {
        self.allowed_types.as_ref()
    }

    /// Returns `true` if the given resource is eligible for this task.
    ///
    /// A resource matches if:
    /// - Its ID is in `allowed_ids`, **or**
    /// - Its type is in `allowed_types`, **or**
    /// - Both sets are `None` (no restriction).
    pub fn matches(&self, resource_id: &str, resource_type: &str) -> bool {
        let id_match = self
            .allowed_ids
            .as_ref()
            .is_some_and(|ids| ids.contains(resource_id));
        let type_match = self
            .allowed_types
            .as_ref()
            .is_some_and(|types| types.contains(resource_type));

        match (&self.allowed_ids, &self.allowed_types) {
            (None, None) => true,
            _ => id_match || type_match,
        }
    }
}

impl fmt::Display for ResourceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.allowed_ids, &self.allowed_types) {
            (Some(ids), Some(types)) => {
                write!(f, "Resource(ids={:?}, types={:?})", ids, types)
            }
            (Some(ids), None) => write!(f, "Resource(ids={:?})", ids),
            (None, Some(types)) => write!(f, "Resource(types={:?})", types),
            (None, None) => write!(f, "Resource(any)"),
        }
    }
}

/// `ResourceConstraint` is not time-dependent — it always returns the full range.
///
/// The actual resource filtering is done by the prescheduler, which reads
/// `allowed_ids`/`allowed_types` to decide which (resource, task) pairs to evaluate.
impl<U: Unit> Constraint<U> for ResourceConstraint {
    fn compute_intervals(&self, range: Interval<U>) -> IntervalSet<U> {
        IntervalSet::from(range)
    }

    fn stringify(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qtty::Second;

    #[test]
    fn from_ids_matches_by_id() {
        let c = ResourceConstraint::from_ids(["LST1", "LST2"]);
        assert!(c.matches("LST1", "anything"));
        assert!(c.matches("LST2", "anything"));
        assert!(!c.matches("MAGIC1", "anything"));
    }

    #[test]
    fn from_types_matches_by_type() {
        let c = ResourceConstraint::from_types(["LST"]);
        assert!(c.matches("LST1", "LST"));
        assert!(c.matches("LST99", "LST"));
        assert!(!c.matches("MAGIC1", "MAGIC"));
    }

    #[test]
    fn from_ids_and_types_matches_union() {
        let c = ResourceConstraint::from_ids_and_types(["LST1"], ["MAGIC"]);
        assert!(c.matches("LST1", "LST")); // ID match
        assert!(c.matches("MAGIC2", "MAGIC")); // type match
        assert!(!c.matches("OTHER", "OTHER")); // neither
    }

    #[test]
    fn unconstrained_matches_everything() {
        let c = ResourceConstraint::new(None, None);
        assert!(c.matches("anything", "anything"));
    }

    #[test]
    fn compute_intervals_returns_full_range() {
        let c = ResourceConstraint::from_ids(["LST1"]);
        let range = Interval::<Second>::from_f64(0.0, 100.0);
        let result = c.compute_intervals(range);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], range);
    }
}
