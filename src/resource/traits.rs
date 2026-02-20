//! Resource trait definition.

use std::fmt::Debug;

use crate::algorithms::rl::types::AgentType;
use crate::constraints::{Constraint, ConstraintExpr};
use crate::solution_space::{Interval, IntervalSet};
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
/// use virolai::resource::Resource;
/// use virolai::solution_space::Interval;
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
    fn compute_availability(&self, horizon: Interval<A>) -> IntervalSet<A> {
        match self.constraints() {
            Some(constraint_tree) => constraint_tree.compute_intervals(horizon),
            None => IntervalSet::from(horizon),
        }
    }

    // --- RL scheduling extensions (all optional, backward-compatible) ---

    /// Returns the agent type (age group) of this resource, if it acts as a mobile agent.
    ///
    /// Used by the RL environment to determine movement speed and to check
    /// whether the resource satisfies task coalition requirements.
    fn agent_type(&self) -> Option<AgentType> {
        None
    }

    /// Returns the 2D spatial position `(x, y)` of this resource, if applicable.
    ///
    /// Used by the RL environment for distance computations and observation encoding.
    fn position_2d(&self) -> Option<(f64, f64)> {
        None
    }

    /// Returns the maximum movement speed of this resource per time step.
    ///
    /// Defaults to 0.0 (stationary). Override when the resource represents
    /// a mobile agent in the RL environment.
    fn max_speed(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use qtty::Second;

    #[derive(Debug)]
    struct UnconstrainedResource;

    impl Resource<Second> for UnconstrainedResource {
        type ConstraintLeaf = IntervalConstraint<Second>;
        fn name(&self) -> &str {
            "unconstrained"
        }
        fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
            None
        }
    }

    #[derive(Debug)]
    struct ConstrainedResource {
        constraints: ConstraintExpr<IntervalConstraint<Second>>,
    }

    impl Resource<Second> for ConstrainedResource {
        type ConstraintLeaf = IntervalConstraint<Second>;
        fn name(&self) -> &str {
            "constrained"
        }
        fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
            Some(&self.constraints)
        }
    }

    #[test]
    fn unconstrained_returns_full_horizon() {
        let r = UnconstrainedResource;
        let horizon = Interval::from_f64(0.0, 100.0);
        let avail = r.compute_availability(horizon);
        assert_eq!(avail.len(), 1);
        assert_eq!(avail[0], horizon);
    }

    #[test]
    fn unconstrained_name() {
        let r = UnconstrainedResource;
        assert_eq!(r.name(), "unconstrained");
    }

    #[test]
    fn constrained_delegates_to_constraint_tree() {
        let constraint = IntervalConstraint::new(Interval::from_f64(20.0, 80.0));
        let r = ConstrainedResource {
            constraints: ConstraintExpr::leaf(constraint),
        };

        let horizon = Interval::from_f64(0.0, 100.0);
        let avail = r.compute_availability(horizon);
        assert_eq!(avail.len(), 1);
        assert_eq!(avail[0], Interval::from_f64(20.0, 80.0));
    }

    #[test]
    fn constrained_with_intersection() {
        let a = IntervalConstraint::new(Interval::from_f64(0.0, 60.0));
        let b = IntervalConstraint::new(Interval::from_f64(40.0, 100.0));
        let r = ConstrainedResource {
            constraints: ConstraintExpr::intersection(vec![
                ConstraintExpr::leaf(a),
                ConstraintExpr::leaf(b),
            ]),
        };

        let horizon = Interval::from_f64(0.0, 100.0);
        let avail = r.compute_availability(horizon);
        assert_eq!(avail.len(), 1);
        assert_eq!(avail[0], Interval::from_f64(40.0, 60.0));
    }
}
