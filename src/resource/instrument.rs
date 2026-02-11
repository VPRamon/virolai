//! Instrument abstraction for shared scheduling constraints.
//!
//! An Instrument wraps resource-level constraints that apply to all tasks
//! scheduled on it. This provides a concrete implementation of the `Resource`
//! trait for common use cases.

use std::fmt::Debug;

use crate::constraints::{Constraint, ConstraintExpr};
use crate::resource::Resource;
use crate::Id;
use qtty::Unit;

/// A generic instrument with shared constraints.
///
/// Instrument provides a concrete implementation of the `Resource` trait,
/// holding a constraint tree that applies to all tasks scheduled on it.
/// Each instrument is assigned a unique auto-generated ID upon creation;
/// users identify instruments by their human-readable `name`.
///
/// # Type Parameters
///
/// - `A`: The axis unit for scheduling (e.g., Day, Second)
/// - `C`: The constraint leaf type (must implement `Constraint<A>`)
///
/// # Example
///
/// ```ignore
/// use vrolai::resource::Instrument;
/// use qtty::Day;
///
/// let telescope = Instrument::new("La Palma Telescope")
///     .with_constraint(nighttime_constraint_tree);
///
/// // Each instrument has an auto-generated unique ID
/// println!("ID: {}, Name: {}", telescope.id(), telescope.name());
///
/// // Compute availability once
/// let availability = telescope.compute_availability(schedule_horizon);
/// ```
#[derive(Debug, Clone)]
pub struct Instrument<A, C>
where
    A: Unit,
    C: Constraint<A>,
{
    /// Unique auto-generated identifier for this instrument
    id: Id,
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
    /// Creates a new instrument with the given name and an auto-generated unique ID.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: crate::generate_id(),
            name: name.into(),
            constraint: None,
            _axis: std::marker::PhantomData,
        }
    }

    /// Returns the unique auto-generated identifier for this instrument.
    pub fn id(&self) -> &str {
        &self.id
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

    fn name(&self) -> &str {
        &self.name
    }

    fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
        self.constraint.as_ref()
    }
}
