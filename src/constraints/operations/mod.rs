mod complement;
mod intersection;
mod union;

pub use complement::compute_complement;
pub use intersection::compute_intersection;
pub use union::compute_union;

#[cfg(debug_assertions)]
pub mod assertions;

#[cfg(not(debug_assertions))]
pub mod assertions {
    use crate::solution_space::Interval;
    use qtty::Unit;

    pub fn is_canonical<U: Unit>(_intervals: &[Interval<U>]) -> bool {
        true
    }
}
