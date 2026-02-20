//! virolai - Virtual Resource Optimization Leveraging AI
//!
//! A constraint-based task scheduling library supporting dependency graphs,
//! solution spaces, and prescheduling utilities.

pub mod algorithms;
pub mod constraints;
pub mod resource;
pub mod schedule;
pub mod scheduling_block;
pub mod solution_space;
pub mod units;

#[cfg(test)]
pub(crate) mod test_utils;

// Re-export unit conversion traits for ergonomic use
pub use units::{convert, SameDim};

// Re-export multi-dimensional solution space types
pub use solution_space::{Axes, Region2, Region3, Region4, SolutionSpaceND};

/// Identifier type used for tasks, resources, and scheduling artifacts.
pub type Id = String;

/// Generates a new unique identifier (UUID v4).
pub fn generate_id() -> Id {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_id_returns_non_empty() {
        let id = generate_id();
        assert!(!id.is_empty());
    }

    #[test]
    fn generate_id_unique() {
        let id1 = generate_id();
        let id2 = generate_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn generate_id_is_valid_uuid() {
        let id = generate_id();
        // UUID v4 format: 8-4-4-4-12 hex chars
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().filter(|c| *c == '-').count(), 4);
    }
}
