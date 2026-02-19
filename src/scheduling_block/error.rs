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

    #[error("Task ID already exists: {0}")]
    DuplicateId(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_detected_display() {
        let e = SchedulingError::CycleDetected;
        assert_eq!(
            e.to_string(),
            "Adding this dependency would create a cycle in the scheduling graph"
        );
    }

    #[test]
    fn invalid_node_index_display() {
        let e = SchedulingError::InvalidNodeIndex(NodeIndex::new(5));
        let s = e.to_string();
        assert!(s.contains("Invalid node index"));
    }

    #[test]
    fn graph_contains_cycle_display() {
        let e = SchedulingError::GraphContainsCycle;
        assert_eq!(
            e.to_string(),
            "Cannot compute topological order: graph contains a cycle"
        );
    }

    #[test]
    fn empty_graph_display() {
        let e = SchedulingError::EmptyGraph;
        assert_eq!(
            e.to_string(),
            "Cannot perform operation: scheduling block is empty"
        );
    }

    #[test]
    fn duplicate_id_display() {
        let e = SchedulingError::DuplicateId("my-task".to_string());
        assert_eq!(e.to_string(), "Task ID already exists: my-task");
    }

    #[test]
    fn error_equality() {
        assert_eq!(
            SchedulingError::CycleDetected,
            SchedulingError::CycleDetected
        );
        assert_ne!(SchedulingError::CycleDetected, SchedulingError::EmptyGraph);
    }
}
