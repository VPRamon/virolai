//! Instrument abstraction for shared scheduling constraints.
//!
//! An Instrument wraps resource-level constraints that apply to all tasks
//! scheduled on it. This provides a concrete implementation of the `Resource`
//! trait for common use cases.

use std::fmt::Debug;

use crate::constraints::{Constraint, ConstraintExpr};
use crate::scheduling_block::Resource;
use qtty::Unit;

/// A generic instrument with shared constraints.
///
/// Instrument provides a concrete implementation of the `Resource` trait,
/// holding a constraint tree that applies to all tasks scheduled on it.
///
/// # Type Parameters
///
/// - `A`: The axis unit for scheduling (e.g., Day, Second)
/// - `C`: The constraint leaf type (must implement `Constraint<A>`)
///
/// # Example
///
/// ```ignore
/// use vrolai::scheduling_block::Instrument;
/// use qtty::Day;
///
/// let telescope = Instrument::new("LST1", "La Palma Telescope")
///     .with_constraint(nighttime_constraint_tree);
///
/// // Compute availability once
/// let availability = telescope.compute_availability(schedule_horizon);
///
/// // Intersect with each task's windows
/// for task in tasks {
///     let task_windows = task.compute_windows(schedule_horizon);
///     let effective_windows = intersect_intervals(task_windows, &availability);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Instrument<A, C>
where
    A: Unit,
    C: Constraint<A>,
{
    /// Unique identifier for this instrument
    id: String,
    /// Human-readable name
    name: String,
    /// Shared constraints (nighttime, moon altitude, etc.)
    constraint: Option<ConstraintExpr<C>>,
    /// Phantom marker for the axis unit
    _axis: std::marker::PhantomData<A>,
}

impl<A, C> Instrument<A, C>
where
    A: Unit,
    C: Constraint<A>,
{
    /// Creates a new instrument with the given ID and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            constraint: None,
            _axis: std::marker::PhantomData,
        }
    }

    /// Adds a constraint tree to this instrument.
    pub fn with_constraint(mut self, constraint: ConstraintExpr<C>) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Sets or replaces the constraint tree.
    pub fn set_constraint(&mut self, constraint: Option<ConstraintExpr<C>>) {
        self.constraint = constraint;
    }

    /// Returns the constraint tree reference.
    pub fn constraint(&self) -> Option<&ConstraintExpr<C>> {
        self.constraint.as_ref()
    }
}

impl<A, C> Resource<A> for Instrument<A, C>
where
    A: Unit + Send + Sync + 'static,
    C: Constraint<A> + 'static,
{
    type ConstraintLeaf = C;

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
        self.constraint.as_ref()
    }
}
