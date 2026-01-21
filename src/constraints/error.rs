use thiserror::Error;

/// Errors that can occur during constraint tree operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    #[error("Cannot add child to a leaf node")]
    CannotAddChildToLeaf,

    #[error("Cannot add child to a NOT node")]
    CannotAddChildToNot,
}
