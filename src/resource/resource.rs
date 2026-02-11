//! Resource trait definition.

use std::fmt::Debug;

use crate::constraints::{Constraint, ConstraintExpr};
use crate::solution_space::Interval;
use qtty::Unit;

/// A schedulable resource with constraints that apply to all tasks using it.
///
/// Resources represent shared scheduling constraints computed once and applied
/// to all tasks. Typical examples include:
/// - Telescope availability (nighttime, weather constraints)
/// - Resource operational limits (moon altitude, sun distance)
/// - Observatory maintenance windows
///
/// # Type Parameters
///
/// - `A`: The **axis unit** used for scheduling intervals (e.g., MJD days).
///
/// # Associated Types
///
/// - `ConstraintLeaf`: The leaf type used in constraint trees. Must implement `Constraint<A>`.
///
/// # Usage Pattern
///
/// 1. Compute resource availability windows once: `resource.compute_availability(horizon)`
/// 2. For each task, intersect task windows with resource windows
/// 3. Schedule within the intersection
///
/// # Example
///
/// ```ignore
/// use vrolai::resource::Resource;
/// use vrolai::solution_space::Interval;
/// use qtty::Day;
///
/// struct Telescope {
///     name: String,
///     constraint: ConstraintExpr<MyConstraintLeaf>,
/// }
///
/// impl Resource<Day> for Telescope {
///     type ConstraintLeaf = MyConstraintLeaf;
///
///     fn name(&self) -> &str { &self.name }
///
///     fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
///         Some(&self.constraint)
///     }
/// }
/// ```
pub trait Resource<A: Unit>: Send + Sync + Debug + 'static {
    /// The leaf constraint type used in constraint trees.
    type ConstraintLeaf: Constraint<A>;

    /// Returns a human-readable name for this resource.
    fn name(&self) -> &str;

    /// Returns the resource-level constraints, if any.
    ///
    /// These constraints are computed once and intersected with all task windows
    /// that use this resource.
    fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>>;

    /// Computes the availability windows for this resource within the given horizon.
    ///
    /// This method evaluates the resource's constraints and returns the intervals
    /// where the resource is available for scheduling.
    ///
    /// # Arguments
    ///
    /// * `horizon` - The scheduling window to evaluate
    ///
    /// # Returns
    ///
    /// A vector of non-overlapping intervals where the resource is available.
    /// Returns a single interval covering the entire horizon if no constraints exist.
    fn compute_availability(&self, horizon: Interval<A>) -> Vec<Interval<A>> {
        match self.constraints() {
            Some(constraint_tree) => constraint_tree.compute_intervals(horizon),
            None => vec![horizon],
        }
    }
}
