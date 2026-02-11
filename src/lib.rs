//! v_rolai - Versatile Instrumentation Resource Optimization Leveraging AI
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

// Re-export unit conversion traits for ergonomic use
pub use units::{convert, SameDim};

/// Identifier type used for tasks, resources, and scheduling artifacts.
pub type Id = String;

/// Generates a new unique identifier (UUID v4).
pub fn generate_id() -> Id {
    uuid::Uuid::new_v4().to_string()
}
