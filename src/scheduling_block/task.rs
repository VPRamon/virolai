use std::fmt::Debug;

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
/// use vrolai::scheduling_block::Task;
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

    /// Returns the required delay after this task completes.
    ///
    /// This delay is added to the cursor when advancing the scheduling timeline,
    /// ensuring proper spacing between tasks (e.g., telescope reconfiguration time).
    ///
    /// Default implementation returns zero (no delay).
    fn delay_after(&self) -> Quantity<A> {
        Quantity::new(0.0)
    }

    /// Computes the required delay between this task and a subsequent task.
    ///
    /// This is used in cross-kind ordering comparisons to determine if a flexible
    /// task would block an endangered task due to required inter-task delays.
    ///
    /// # Arguments
    ///
    /// * `previous_task` - The task that would execute before this one
    ///
    /// # Returns
    ///
    /// The required delay in axis units. Default implementation returns zero.
    fn compute_delay_after(&self, _previous_task: &Self) -> Quantity<A> {
        Quantity::new(0.0)
    }
}
