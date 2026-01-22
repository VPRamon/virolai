use super::error::SchedulingError;
use super::task::Task;
use petgraph::algo::{has_path_connecting, toposort};
use petgraph::stable_graph::StableGraph;
use petgraph::{Directed, Direction, EdgeType};
use qtty::{Quantity, Second, Unit};
use std::fmt::Display;

/// DAG-based task scheduler with dependency tracking.
///
/// # Invariants
///
/// - The dependency graph must remain acyclic; [`add_dependency`](Self::add_dependency) enforces this
/// - Tasks are identified by stable `NodeIndex` handles that persist across removals
/// - Dependencies represent ordering constraints: task `from` must complete before `to`
///
/// # Example
///
/// ```ignore
/// let mut block = SchedulingBlock::new();
/// let task_a = block.add_task(my_task_a);
/// let task_b = block.add_task(my_task_b);
/// block.add_dependency(task_a, task_b, ()).expect("no cycle");
/// ```
#[derive(Debug, Clone)]
pub struct SchedulingBlock<T, U = Second, D = (), E = Directed>
where
    T: Task<U>,
    U: Unit,
    E: EdgeType,
{
    graph: StableGraph<T, D, E>,
    _phantom: std::marker::PhantomData<U>,
}

impl<T, U, D, E> Default for SchedulingBlock<T, U, D, E>
where
    T: Task<U>,
    U: Unit,
    E: EdgeType,
{
    fn default() -> Self {
        Self {
            graph: StableGraph::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, U, D, E> SchedulingBlock<T, U, D, E>
where
    T: Task<U>,
    U: Unit,
    E: EdgeType,
{
    pub fn new() -> Self {
        Self {
            graph: StableGraph::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn add_task(&mut self, task: T) -> petgraph::graph::NodeIndex {
        self.graph.add_node(task)
    }

    /// Adds dependency edge `from` → `to`.
    ///
    /// # Errors
    ///
    /// - `InvalidNodeIndex` if either node does not exist
    /// - `CycleDetected` if adding the edge would create a cycle
    pub fn add_dependency(
        &mut self,
        from: petgraph::graph::NodeIndex,
        to: petgraph::graph::NodeIndex,
        dep: D,
    ) -> Result<(), SchedulingError> {
        if !self.graph.contains_node(from) {
            return Err(SchedulingError::InvalidNodeIndex(from));
        }
        if !self.graph.contains_node(to) {
            return Err(SchedulingError::InvalidNodeIndex(to));
        }

        // Cycle check: edge from→to creates cycle if path to→from already exists
        if has_path_connecting(&self.graph, to, from, None) {
            return Err(SchedulingError::CycleDetected);
        }

        self.graph.add_edge(from, to, dep);
        Ok(())
    }

    /// Returns task nodes in topological order.
    ///
    /// # Errors
    ///
    /// Returns `GraphContainsCycle` if the graph has a cycle (should never occur if
    /// [`add_dependency`](Self::add_dependency) is used correctly).
    pub fn topo_order(&self) -> Result<Vec<petgraph::graph::NodeIndex>, SchedulingError> {
        toposort(&self.graph, None).map_err(|_| SchedulingError::GraphContainsCycle)
    }

    /// Returns tasks with no predecessors (entry points).
    pub fn roots(&self) -> Vec<petgraph::graph::NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&node| {
                self.graph
                    .neighbors_directed(node, Direction::Incoming)
                    .count()
                    == 0
            })
            .collect()
    }

    /// Returns tasks with no successors (exit points).
    pub fn leaves(&self) -> Vec<petgraph::graph::NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&node| {
                self.graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .count()
                    == 0
            })
            .collect()
    }

    pub fn task_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn dependency_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns reference to the underlying graph.
    pub fn graph(&self) -> &StableGraph<T, D, E> {
        &self.graph
    }

    /// Returns mutable access to the graph.
    ///
    /// # Safety
    ///
    /// Direct mutations bypass cycle detection. Ensure no cycles are introduced.
    pub fn graph_mut(&mut self) -> &mut StableGraph<T, D, E> {
        &mut self.graph
    }

    pub fn get_task(&self, node: petgraph::graph::NodeIndex) -> Option<&T> {
        self.graph.node_weight(node)
    }

    pub fn get_task_mut(&mut self, node: petgraph::graph::NodeIndex) -> Option<&mut T> {
        self.graph.node_weight_mut(node)
    }

    pub fn predecessors(
        &self,
        node: petgraph::graph::NodeIndex,
    ) -> Vec<petgraph::graph::NodeIndex> {
        self.graph
            .neighbors_directed(node, Direction::Incoming)
            .collect()
    }

    pub fn successors(&self, node: petgraph::graph::NodeIndex) -> Vec<petgraph::graph::NodeIndex> {
        self.graph
            .neighbors_directed(node, Direction::Outgoing)
            .collect()
    }
}

// Implementation of time-dependent methods (generic over unit)
impl<T, U, D, E> SchedulingBlock<T, U, D, E>
where
    T: Task<U>,
    U: Unit,
    E: EdgeType,
{
    /// Computes critical path: longest chain through the dependency graph.
    ///
    /// Returns total duration (in axis units) and the sequence of nodes
    /// on the critical path.
    ///
    /// # Errors
    ///
    /// Returns `EmptyGraph` if there are no tasks.
    pub fn critical_path(&self) -> Result<(f64, Vec<petgraph::graph::NodeIndex>), SchedulingError> {
        if self.graph.node_count() == 0 {
            return Err(SchedulingError::EmptyGraph);
        }

        let topo = self.topo_order()?;
        let mut earliest_start = vec![0.0_f64; self.graph.node_count()];
        let mut predecessor = vec![None; self.graph.node_count()];

        for &node in &topo {
            let node_idx = node.index();
            // Use size_on_axis() for scheduling math
            let task_duration = self.graph[node].size_on_axis().value();

            for successor in self.successors(node) {
                let succ_idx = successor.index();
                let new_start = earliest_start[node_idx] + task_duration;

                if new_start > earliest_start[succ_idx] {
                    earliest_start[succ_idx] = new_start;
                    predecessor[succ_idx] = Some(node);
                }
            }
        }

        let mut max_finish = 0.0_f64;
        let mut end_node = None;

        for node in self.graph.node_indices() {
            let node_idx = node.index();
            // Use size_on_axis() for scheduling math
            let finish_time = earliest_start[node_idx] + self.graph[node].size_on_axis().value();

            if finish_time > max_finish {
                max_finish = finish_time;
                end_node = Some(node);
            }
        }

        let mut path = Vec::new();
        let mut current = end_node;

        while let Some(node) = current {
            path.push(node);
            current = predecessor[node.index()];
        }

        path.reverse();
        Ok((max_finish, path))
    }
}

impl<T, U, D, E> Display for SchedulingBlock<T, U, D, E>
where
    T: Task<U>,
    U: Unit,
    E: EdgeType,
    Quantity<U>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SchedulingBlock {{")?;
        writeln!(f, "  Tasks: {}", self.graph.node_count())?;
        writeln!(f, "  Dependencies: {}", self.graph.edge_count())?;

        if self.graph.node_count() > 0 {
            writeln!(f, "  Task list:")?;
            for node in self.graph.node_indices() {
                let task = &self.graph[node];
                writeln!(
                    f,
                    "    [{}] {} (size: {}, priority: {})",
                    node.index(),
                    task.name(),
                    task.size_on_axis(),
                    task.priority()
                )?;
            }
        }

        write!(f, "}}")
    }
}
