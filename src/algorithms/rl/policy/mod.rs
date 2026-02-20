//! Policy trait and implementations.

pub mod heuristic;
pub mod random;
pub mod trait_;

pub use heuristic::GreedyHeuristicPolicy;
pub use random::RandomPolicy;
pub use trait_::Policy;
