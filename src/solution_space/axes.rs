//! The `Axes` trait and multi-dimensional region types.
//!
//! An **axes** type describes the product of physical dimensions that make up
//! the scheduling space.  The most common case is 1-D (a single time axis),
//! but any tuple of [`qtty::Unit`] types from 1 to 4 can be used.
//!
//! # How it works
//!
//! Each tuple `(U1,)`, `(U1, U2)`, … provides:
//!
//! - **`Primary`** — the unit of the first axis.  The EST engine always
//!   operates on this axis; all 1-D scheduling algorithms are unchanged.
//! - **`Region`** — the concrete type that stores per-axis feasibility data
//!   for a single task.  For `(U,)` this is just `IntervalSet<U>`, so
//!   existing code needs no changes.  For higher arities it is one of the
//!   `Region2 / Region3 / Region4` structs below.
//!
//! # Backward compatibility
//!
//! The type alias `SolutionSpace<U> = SolutionSpaceND<(U,)>` keeps every
//! existing call site compiling without changes.

use std::fmt::Debug;

use qtty::Unit;

use super::interval_set::IntervalSet;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A product of physical dimensions that parametrises a [`SolutionSpaceND`].
///
/// Implementors are tuples of [`Unit`] types such as `(Second,)`,
/// `(Second, Hertz)`, or `(Day, Meter, Hertz)`.
///
/// # Provided implementations
///
/// | Tuple            | Arity | `Region` type              |
/// |------------------|-------|----------------------------|
/// | `(U,)`           | 1-D   | `IntervalSet<U>`           |
/// | `(U1, U2)`       | 2-D   | [`Region2<U1, U2>`]        |
/// | `(U1, U2, U3)`   | 3-D   | [`Region3<U1, U2, U3>`]    |
/// | `(U1, U2, U3, U4)` | 4-D | [`Region4<U1, U2, U3, U4>`] |
///
/// [`SolutionSpaceND`]: crate::solution_space::SolutionSpaceND
pub trait Axes {
    /// The primary scheduling axis (axis-0).
    ///
    /// All EST metrics, candidates, and the scheduling engine operate
    /// exclusively on this axis.  For a 1-D setup this equals the sole axis.
    type Primary: Unit;

    /// The type that represents a feasibility region for all axes of this
    /// product space for a single task.
    ///
    /// Must be `Clone + Debug`.
    type Region: Clone + Debug;

    /// Returns a reference to the primary-axis interval set stored inside
    /// a region.
    fn primary_of(region: &Self::Region) -> &IntervalSet<Self::Primary>;

    /// Constructs a region whose primary axis is given by `primary` and
    /// whose secondary axes are initialised to **empty** (unconstrained —
    /// callers should fill them in via [`SolutionSpaceND::get_region_mut`]
    /// if they want per-axis feasibility data).
    fn region_from_primary(primary: IntervalSet<Self::Primary>) -> Self::Region;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D: (U,)
// ─────────────────────────────────────────────────────────────────────────────

/// `(U,)` is the 1-D axes type.  Its `Region` is plain `IntervalSet<U>`,
/// which means `SolutionSpace<U> = SolutionSpaceND<(U,)>` is a perfect
/// drop-in replacement for the previous concrete struct.
impl<U: Unit> Axes for (U,) {
    type Primary = U;
    type Region = IntervalSet<U>;

    #[inline]
    fn primary_of(region: &IntervalSet<U>) -> &IntervalSet<U> {
        region
    }

    #[inline]
    fn region_from_primary(primary: IntervalSet<U>) -> IntervalSet<U> {
        primary
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D: (U1, U2)
// ─────────────────────────────────────────────────────────────────────────────

/// Feasibility region for a 2-dimensional separable product space.
///
/// A task is feasible if *both* axes independently admit the required
/// footprint.  The axes are stored as independent `IntervalSet`s (separable
/// constraints); non-separable regions (e.g. polytopes) are out of scope for
/// the current version.
///
/// # Example
///
/// ```rust
/// use virolai::solution_space::{Axes, Region2, SolutionSpaceND};
/// use virolai::solution_space::IntervalSet;
/// use qtty::{Meter, Second};
///
/// // Build a 2-D space (time × distance).
/// let mut space: SolutionSpaceND<(Second, Meter)> = SolutionSpaceND::new();
/// ```
#[derive(Debug, Clone)]
pub struct Region2<U1: Unit, U2: Unit> {
    /// Feasible intervals on the primary axis (index 0).
    pub axis_0: IntervalSet<U1>,
    /// Feasible intervals on the secondary axis (index 1).
    pub axis_1: IntervalSet<U2>,
}

impl<U1: Unit, U2: Unit> Region2<U1, U2> {
    /// Creates an empty 2-D region (all axes unconstrained).
    pub fn empty() -> Self {
        Self {
            axis_0: IntervalSet::new(),
            axis_1: IntervalSet::new(),
        }
    }

    /// Creates a region from per-axis interval sets.
    pub fn new(axis_0: IntervalSet<U1>, axis_1: IntervalSet<U2>) -> Self {
        Self { axis_0, axis_1 }
    }
}

impl<U1: Unit, U2: Unit> Axes for (U1, U2) {
    type Primary = U1;
    type Region = Region2<U1, U2>;

    #[inline]
    fn primary_of(region: &Region2<U1, U2>) -> &IntervalSet<U1> {
        &region.axis_0
    }

    fn region_from_primary(primary: IntervalSet<U1>) -> Region2<U1, U2> {
        Region2 {
            axis_0: primary,
            axis_1: IntervalSet::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D: (U1, U2, U3)
// ─────────────────────────────────────────────────────────────────────────────

/// Feasibility region for a 3-dimensional separable product space.
#[derive(Debug, Clone)]
pub struct Region3<U1: Unit, U2: Unit, U3: Unit> {
    /// Feasible intervals on the primary axis (index 0).
    pub axis_0: IntervalSet<U1>,
    /// Feasible intervals on axis index 1.
    pub axis_1: IntervalSet<U2>,
    /// Feasible intervals on axis index 2.
    pub axis_2: IntervalSet<U3>,
}

impl<U1: Unit, U2: Unit, U3: Unit> Region3<U1, U2, U3> {
    /// Creates an empty 3-D region.
    pub fn empty() -> Self {
        Self {
            axis_0: IntervalSet::new(),
            axis_1: IntervalSet::new(),
            axis_2: IntervalSet::new(),
        }
    }

    /// Creates a region from per-axis interval sets.
    pub fn new(axis_0: IntervalSet<U1>, axis_1: IntervalSet<U2>, axis_2: IntervalSet<U3>) -> Self {
        Self {
            axis_0,
            axis_1,
            axis_2,
        }
    }
}

impl<U1: Unit, U2: Unit, U3: Unit> Axes for (U1, U2, U3) {
    type Primary = U1;
    type Region = Region3<U1, U2, U3>;

    #[inline]
    fn primary_of(region: &Region3<U1, U2, U3>) -> &IntervalSet<U1> {
        &region.axis_0
    }

    fn region_from_primary(primary: IntervalSet<U1>) -> Region3<U1, U2, U3> {
        Region3 {
            axis_0: primary,
            axis_1: IntervalSet::new(),
            axis_2: IntervalSet::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4-D: (U1, U2, U3, U4)
// ─────────────────────────────────────────────────────────────────────────────

/// Feasibility region for a 4-dimensional separable product space.
#[derive(Debug, Clone)]
pub struct Region4<U1: Unit, U2: Unit, U3: Unit, U4: Unit> {
    /// Feasible intervals on the primary axis (index 0).
    pub axis_0: IntervalSet<U1>,
    /// Feasible intervals on axis index 1.
    pub axis_1: IntervalSet<U2>,
    /// Feasible intervals on axis index 2.
    pub axis_2: IntervalSet<U3>,
    /// Feasible intervals on axis index 3.
    pub axis_3: IntervalSet<U4>,
}

impl<U1: Unit, U2: Unit, U3: Unit, U4: Unit> Region4<U1, U2, U3, U4> {
    /// Creates an empty 4-D region.
    pub fn empty() -> Self {
        Self {
            axis_0: IntervalSet::new(),
            axis_1: IntervalSet::new(),
            axis_2: IntervalSet::new(),
            axis_3: IntervalSet::new(),
        }
    }

    /// Creates a region from per-axis interval sets.
    pub fn new(
        axis_0: IntervalSet<U1>,
        axis_1: IntervalSet<U2>,
        axis_2: IntervalSet<U3>,
        axis_3: IntervalSet<U4>,
    ) -> Self {
        Self {
            axis_0,
            axis_1,
            axis_2,
            axis_3,
        }
    }
}

impl<U1: Unit, U2: Unit, U3: Unit, U4: Unit> Axes for (U1, U2, U3, U4) {
    type Primary = U1;
    type Region = Region4<U1, U2, U3, U4>;

    #[inline]
    fn primary_of(region: &Region4<U1, U2, U3, U4>) -> &IntervalSet<U1> {
        &region.axis_0
    }

    fn region_from_primary(primary: IntervalSet<U1>) -> Region4<U1, U2, U3, U4> {
        Region4 {
            axis_0: primary,
            axis_1: IntervalSet::new(),
            axis_2: IntervalSet::new(),
            axis_3: IntervalSet::new(),
        }
    }
}
