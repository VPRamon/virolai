use petgraph::graph::NodeIndex;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SchedulingError {
    #[error("Adding this dependency would create a cycle in the scheduling graph")]
    CycleDetected,

    #[error("Invalid node index: {0:?}")]
    InvalidNodeIndex(NodeIndex),

    #[error("Cannot compute topological order: graph contains a cycle")]
    GraphContainsCycle,

    #[error("Cannot perform operation: scheduling block is empty")]
    EmptyGraph,
}
