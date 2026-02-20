//! N-dimensional task trait.
//!
//! [`TaskND<A>`] is the multi-axis generalisation of [`Task<U>`].  Every type
//! that already implements `Task<U>` automatically satisfies
//! `TaskND<(U,)>` through a blanket impl, so **existing user code requires no
//! changes**.
//!
//! New task types intended for multi-axis solution spaces can implement
//! `TaskND<(U1, U2)>` (or higher arities) directly.
//!
//! [`Task<U>`]: super::task::Task

use std::fmt::Debug;

use crate::solution_space::Axes;
use qtty::{Quantity, Unit};

use super::task::Task;

// ─────────────────────────────────────────────────────────────────────────────
// Trait definition
// ─────────────────────────────────────────────────────────────────────────────

/// Schedulable task that may span multiple physical axes.
///
/// # Type parameter
///
/// `A: Axes` encodes the product of dimensions that the task lives in.
/// For a classic 1-D scheduling setup use `A = (Second,)` (or whichever
/// time unit you prefer).
///
/// # Blanket implementation
///
/// Any type that implements [`Task<U>`] also implements `TaskND<(U,)>` —
/// no boilerplate required for the 1-D case.
///
/// # Example – 2-D task
///
/// ```rust
/// use virolai::solution_space::{Axes, Region2, SolutionSpaceND};
/// use virolai::scheduling_block::TaskND;
/// use qtty::{Meter, Quantity, Second};
/// use std::fmt::Debug;
///
/// #[derive(Debug)]
/// struct Observation {
///     duration: Quantity<Second>,
///     range: Quantity<Meter>,
/// }
///
/// impl TaskND<(Second, Meter)> for Observation {
///     fn name(&self) -> &str { "Observation" }
///     fn size_on_primary(&self) -> Quantity<Second> { self.duration }
/// }
/// ```
pub trait TaskND<A: Axes>: Send + Sync + Debug + 'static {
    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;

    /// Returns the task's required footprint on the **primary** axis.
    ///
    /// This is the value used by the EST engine for interval arithmetic.
    fn size_on_primary(&self) -> Quantity<A::Primary>;

    /// Scheduling priority.  Higher values are scheduled first.
    /// Defaults to `0`.
    fn priority(&self) -> i32 {
        0
    }

    /// Required gap after the task on the primary axis.  This gap is added
    /// to the cursor when advancing the scheduling timeline.
    /// Defaults to zero.
    fn gap_after(&self) -> Quantity<A::Primary> {
        Quantity::new(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: Task<U> → TaskND<(U,)>
// ─────────────────────────────────────────────────────────────────────────────

impl<T, U> TaskND<(U,)> for T
where
    T: Task<U>,
    U: Unit,
{
    #[inline]
    fn name(&self) -> &str {
        Task::name(self)
    }

    #[inline]
    fn size_on_primary(&self) -> Quantity<U> {
        Task::size_on_axis(self)
    }

    #[inline]
    fn priority(&self) -> i32 {
        Task::priority(self)
    }

    #[inline]
    fn gap_after(&self) -> Quantity<U> {
        Task::gap_after(self)
    }
}
