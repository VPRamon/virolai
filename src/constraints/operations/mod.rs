mod complement;
mod intersection;
mod union;

pub use complement::compute_complement;
pub use intersection::compute_intersection;
pub use union::compute_union;

#[cfg(debug_assertions)]
pub mod assertions;
