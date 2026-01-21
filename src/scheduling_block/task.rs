use std::fmt::Debug;

use crate::constraints::{Constraint, ConstraintExpr};
use qtty::{Quantity, Unit};

/// Schedulable task with size, optional priority, and constraints.
///
/// # Invariants
///
/// - `size()` must return a positive quantity
/// - `name()` should uniquely identify the task within a scheduling context
/// - `priority()` defaults to 0; higher values indicate greater importance
/// - `constraints()` returns `None` if the task is unconstrained
///
/// # Associated Type
///
/// - `ConstraintLeaf`: The leaf type used in constraint trees. Must implement
///   `Constraint<U>` so the tree can compute intervals.
pub trait Task<U: Unit>: Send + Sync + Debug + 'static {
    /// The leaf constraint type used in constraint trees.
    type ConstraintLeaf: Constraint<U>;

    fn id(&self) -> u64;
    fn name(&self) -> String;
    fn size(&self) -> Quantity<U>;
    fn priority(&self) -> i32 {
        0
    }
    fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
        None
    }
}
