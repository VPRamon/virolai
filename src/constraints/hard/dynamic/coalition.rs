//! Coalition constraint — requires multiple resources to cover a task simultaneously.
//!
//! This is a **hard + dynamic** constraint that checks whether enough resources
//! of each required type have scheduled the same task during overlapping time windows.
//!
//! Unlike static constraints, coalition feasibility depends on runtime scheduling state
//! (which resources have already been assigned to this task).
//!
//! # Design
//!
//! The `CoalitionConstraint` stores a typed requirement map:
//! `HashMap<String, u32>` mapping resource type → minimum count.
//!
//! For example, `{ "LST": 2, "MAGIC": 1 }` means the task needs at least 2 LST
//! telescopes and 1 MAGIC telescope to be assigned to it simultaneously.
//!
//! # Evaluation
//!
//! During scheduling, a multi-resource algorithm can query this constraint to
//! determine whether sufficient resources have been committed.  The `is_satisfied`
//! method is the primary check point.
//!
//! The `DynamicConstraint` implementation is a secondary integration path for
//! algorithms that use the edge-based dynamic constraint evaluation.

use std::collections::HashMap;
use std::fmt;

/// Requires a minimum number of resources per type to be assigned to a task.
///
/// # Example
///
/// ```
/// use virolai::constraints::CoalitionConstraint;
/// use std::collections::HashMap;
///
/// // Task needs 2 LST and 1 MAGIC simultaneously
/// let coalition = CoalitionConstraint::new([("LST", 2), ("MAGIC", 1)]);
///
/// // Check satisfaction given current assignment counts
/// let mut assigned = HashMap::new();
/// assigned.insert("LST".to_string(), 2u32);
/// assigned.insert("MAGIC".to_string(), 1);
/// assert!(coalition.is_satisfied(&assigned));
///
/// // Not enough MAGIC telescopes
/// assigned.insert("MAGIC".to_string(), 0);
/// assert!(!coalition.is_satisfied(&assigned));
/// ```
#[derive(Debug, Clone)]
pub struct CoalitionConstraint {
    /// Maps resource type → minimum count required.
    requirements: HashMap<String, u32>,
}

impl CoalitionConstraint {
    /// Creates a new coalition constraint from type → count pairs.
    pub fn new(requirements: impl IntoIterator<Item = (impl Into<String>, u32)>) -> Self {
        Self {
            requirements: requirements
                .into_iter()
                .map(|(k, v)| (k.into(), v))
                .collect(),
        }
    }

    /// Creates a simple coalition requiring `count` resources of a single type.
    pub fn single_type(resource_type: impl Into<String>, count: u32) -> Self {
        let mut requirements = HashMap::new();
        requirements.insert(resource_type.into(), count);
        Self { requirements }
    }

    /// Returns the requirement map (resource type → minimum count).
    pub fn requirements(&self) -> &HashMap<String, u32> {
        &self.requirements
    }

    /// Total number of resources required across all types.
    pub fn total_required(&self) -> u32 {
        self.requirements.values().sum()
    }

    /// Returns the requirement for a specific resource type.
    ///
    /// Returns 0 if the type is not in the requirements.
    pub fn requirement_for(&self, resource_type: &str) -> u32 {
        self.requirements.get(resource_type).copied().unwrap_or(0)
    }

    /// Returns `true` if the given assignment counts satisfy all requirements.
    ///
    /// `assigned` maps resource type → number of resources of that type currently
    /// assigned to the task.  Any type not present in `assigned` is treated as 0.
    pub fn is_satisfied(&self, assigned: &HashMap<String, u32>) -> bool {
        self.requirements
            .iter()
            .all(|(rtype, &required)| assigned.get(rtype).copied().unwrap_or(0) >= required)
    }

    /// Returns the deficit for each type: how many more resources are needed.
    ///
    /// Only types with a deficit > 0 are included. Empty map means satisfied.
    pub fn deficit(&self, assigned: &HashMap<String, u32>) -> HashMap<String, u32> {
        self.requirements
            .iter()
            .filter_map(|(rtype, &required)| {
                let have = assigned.get(rtype).copied().unwrap_or(0);
                if have < required {
                    Some((rtype.clone(), required - have))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl fmt::Display for CoalitionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coalition(")?;
        let mut first = true;
        for (rtype, count) in &self.requirements {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}×{}", count, rtype)?;
            first = false;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_type_creation() {
        let c = CoalitionConstraint::single_type("LST", 3);
        assert_eq!(c.total_required(), 3);
        assert_eq!(c.requirement_for("LST"), 3);
        assert_eq!(c.requirement_for("MAGIC"), 0);
    }

    #[test]
    fn multi_type_satisfaction() {
        let c = CoalitionConstraint::new([("LST", 2), ("MAGIC", 1)]);

        let mut assigned = HashMap::new();
        assigned.insert("LST".to_string(), 2);
        assigned.insert("MAGIC".to_string(), 1);
        assert!(c.is_satisfied(&assigned));

        // Exceeding is fine
        assigned.insert("LST".to_string(), 5);
        assert!(c.is_satisfied(&assigned));
    }

    #[test]
    fn multi_type_deficit() {
        let c = CoalitionConstraint::new([("LST", 2), ("MAGIC", 1)]);

        let mut assigned = HashMap::new();
        assigned.insert("LST".to_string(), 1);
        // MAGIC not assigned at all

        assert!(!c.is_satisfied(&assigned));
        let deficit = c.deficit(&assigned);
        assert_eq!(deficit.get("LST"), Some(&1));
        assert_eq!(deficit.get("MAGIC"), Some(&1));
    }

    #[test]
    fn empty_assignment_with_requirements() {
        let c = CoalitionConstraint::single_type("LST", 1);
        let assigned = HashMap::new();
        assert!(!c.is_satisfied(&assigned));
    }

    #[test]
    fn no_requirements_always_satisfied() {
        let c = CoalitionConstraint::new(std::iter::empty::<(&str, u32)>());
        let assigned = HashMap::new();
        assert!(c.is_satisfied(&assigned));
    }

    #[test]
    fn display_format() {
        let c = CoalitionConstraint::single_type("LST", 2);
        let s = c.to_string();
        assert!(s.contains("2×LST"), "got: {}", s);
    }
}
