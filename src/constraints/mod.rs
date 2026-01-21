pub mod constraint;
pub mod error;
pub mod node;
pub mod operations;
pub mod tree;

pub use constraint::Constraint;
pub use constraint::IntervalConstraint;
pub use error::ConstraintError;
pub use node::ConstraintNode;

use qtty::{Quantity, Unit};

/// Returns the minimum of two quantities.
pub fn quantity_min<U: Unit>(a: Quantity<U>, b: Quantity<U>) -> Quantity<U> {
    match a.value().partial_cmp(&b.value()) {
        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => a,
        _ => b,
    }
}

/// Returns the maximum of two quantities.
pub fn quantity_max<U: Unit>(a: Quantity<U>, b: Quantity<U>) -> Quantity<U> {
    match a.value().partial_cmp(&b.value()) {
        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => a,
        _ => b,
    }
}
