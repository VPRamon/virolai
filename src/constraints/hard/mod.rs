//! Hard constraints — binary feasibility checks.
//!
//! Subdivided by data lifetime:
//! - [`static_`] — precomputed before the scheduling loop.
//! - [`dynamic`] — evaluated at runtime against mutable state.

pub mod dynamic;
#[allow(non_snake_case)]
pub mod static_;

// Re-export the static API at the `hard` level for convenience.
pub use static_::Constraint;
pub use static_::IntervalConstraint;

// Re-export key dynamic types for ergonomic access.
pub use dynamic::{
    DynConstraintKind, DynamicConstraint, DynamicConstraintIndex, SchedulingContext,
};
