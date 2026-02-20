use std::fmt::Debug;

use crate::algorithms::rl::types::AgentTypeRequirements;
use crate::constraints::{Constraint, ConstraintExpr};
use crate::units::SameDim;
use qtty::{Quantity, Unit};

/// Schedulable task with size, optional priority, and constraints.
///
/// # Type Parameters
///
/// - `A`: The **axis unit** used for scheduling intervals and constraints (e.g., MJD days).
///   This is the unit in which all scheduling math operates.
///
/// # Associated Types
///
/// - `SizeUnit`: The unit in which the task's duration is naturally expressed (e.g., seconds).
///   Must share the same physical dimension as `A` (enforced by `SameDim<A>` bound).
/// - `ConstraintLeaf`: The leaf type used in constraint trees. Must implement `Constraint<A>`.
///
/// # Invariants
///
/// - `size()` must return a positive quantity
/// - `name()` should uniquely identify the task within a scheduling context
/// - `priority()` defaults to 0; higher values indicate greater importance
/// - `constraints()` returns `None` if the task is unconstrained
///
/// # Unit Conversion
///
/// The `size_on_axis()` method converts the task's natural size to the axis unit.
/// This conversion is performed via `Quantity::to::<A>()`, which uses the compile-time
/// ratio between units (a const operation with no runtime overhead beyond the multiply).
///
/// # Example
///
/// ```ignore
/// use qtty::{Day, Second, Quantity};
/// use virolai::scheduling_block::Task;
///
/// struct MyTask {
///     duration: Quantity<Second>, // Natural unit: seconds
/// }
///
/// impl Task<Day> for MyTask {
///     type SizeUnit = Second;
///     type ConstraintLeaf = MyConstraint;
///
///     fn size(&self) -> Quantity<Second> {
///         self.duration
///     }
///
///     // Inherited: size_on_axis() converts seconds → days automatically
/// }
/// ```
pub trait Task<A: Unit>: Send + Sync + Debug + 'static {
    /// The unit in which this task's size is expressed.
    /// Must share the same dimension as the axis unit `A`.
    type SizeUnit: SameDim<A>;

    /// The leaf constraint type used in constraint trees.
    type ConstraintLeaf: Constraint<A>;

    fn name(&self) -> &str;

    /// Returns the task's duration in its natural unit (`SizeUnit`).
    fn size(&self) -> Quantity<Self::SizeUnit>;

    /// Returns the task's duration converted to the axis unit `A`.
    ///
    /// This is the value used by scheduling algorithms for interval arithmetic.
    /// The default implementation converts via `size().to::<A>()`.
    fn size_on_axis(&self) -> Quantity<A> {
        self.size().to::<A>()
    }

    fn priority(&self) -> i32 {
        0
    }
    fn constraints(&self) -> Option<&ConstraintExpr<Self::ConstraintLeaf>> {
        None
    }

    /// Returns the required gap after this task completes.
    ///
    /// This gap is added to the cursor when advancing the scheduling timeline,
    /// ensuring proper spacing between tasks (e.g., reconfiguration clearance).
    ///
    /// Default implementation returns zero (no gap).
    fn gap_after(&self) -> Quantity<A> {
        Quantity::new(0.0)
    }

    /// Computes the required gap between this task and a subsequent task.
    ///
    /// This is used in cross-kind ordering comparisons to determine if a flexible
    /// task would block an endangered task due to required inter-task gaps.
    ///
    /// # Arguments
    ///
    /// * `previous_task` - The task that would execute before this one
    ///
    /// # Returns
    ///
    /// The required gap in axis units. Default implementation returns zero.
    fn compute_gap_after(&self, _previous_task: &Self) -> Quantity<A> {
        Quantity::new(0.0)
    }

    // --- RL scheduling extensions (all optional, backward-compatible) ---

    /// Returns the reward value for collecting this task.
    ///
    /// Used by the RL environment to determine the reward granted when the task
    /// is successfully collected by a qualifying coalition of agents.
    /// Default returns the priority cast to f64.
    fn value(&self) -> f64 {
        self.priority() as f64
    }

    /// Returns the absolute deadline for this task, if any.
    ///
    /// If the task is not collected before this time, it expires and
    /// disappears. `None` means the task never expires within the horizon.
    fn deadline(&self) -> Option<Quantity<A>> {
        None
    }

    /// Returns the 2D spatial position `(x, y)` of this task, if applicable.
    ///
    /// Used by the RL environment for distance computations, coalition
    /// collection radius checks, and observation encoding.
    fn position_2d(&self) -> Option<(f64, f64)> {
        None
    }

    /// Returns the collection radius for this task.
    ///
    /// Agents must be within this distance of the task position for their
    /// presence to count toward coalition requirements. Default is 0.
    fn collection_radius(&self) -> f64 {
        0.0
    }

    /// Returns the minimum agent-type requirements for collecting this task.
    ///
    /// A task is collected when at least the specified number of each agent type
    /// are simultaneously within the collection radius. Excess agents are allowed
    /// and never penalized.
    fn type_requirements(&self) -> Option<AgentTypeRequirements> {
        None
    }

    /// Returns the maximum number of times this task/type can appear in an episode.
    ///
    /// `None` means unlimited appearances.
    fn max_appearances(&self) -> Option<u32> {
        None
    }
}
