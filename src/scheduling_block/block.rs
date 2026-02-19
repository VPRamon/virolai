use super::error::SchedulingError;
use super::task::Task;
use crate::Id;
use petgraph::algo::{has_path_connecting, toposort};
use petgraph::stable_graph::StableGraph;
use petgraph::{Directed, Direction, EdgeType};
use qtty::{Quantity, Second, Unit};
use std::collections::HashMap;
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
    /// Maps node index → auto-generated unique ID.
    id_by_node: HashMap<petgraph::graph::NodeIndex, Id>,
    /// Maps ID → node index for reverse lookup.
    node_by_id: HashMap<Id, petgraph::graph::NodeIndex>,
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
            id_by_node: HashMap::new(),
            node_by_id: HashMap::new(),
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
            id_by_node: HashMap::new(),
            node_by_id: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Adds a task and returns a unique auto-generated ID for it.
    pub fn add_task(&mut self, task: T) -> Id {
        self.add_task_with_id(task, None)
            .expect("auto-generated UUIDs never collide")
    }

    /// Adds a task with a custom ID, or generates one if `None` is provided.
    ///
    /// Returns the ID that was used (either provided or generated).
    ///
    /// # Errors
    ///
    /// Returns [`SchedulingError::DuplicateId`] if the supplied `id` is already
    /// registered in this block.  Auto-generated IDs (when `id` is `None`) are
    /// UUIDs and will never collide in practice.
    pub fn add_task_with_id(&mut self, task: T, id: Option<Id>) -> Result<Id, SchedulingError> {
        let id = id.unwrap_or_else(crate::generate_id);
        if self.node_by_id.contains_key(&id) {
            return Err(SchedulingError::DuplicateId(id));
        }
        let node = self.graph.add_node(task);
        self.id_by_node.insert(node, id.clone());
        self.node_by_id.insert(id.clone(), node);
        Ok(id)
    }

    /// Returns the auto-generated ID for a node index, if it exists.
    pub fn id_of(&self, node: petgraph::graph::NodeIndex) -> Option<&str> {
        self.id_by_node.get(&node).map(|s| s.as_str())
    }

    /// Returns the node index for a task ID, if it exists.
    pub fn node_of(&self, id: &str) -> Option<petgraph::graph::NodeIndex> {
        self.node_by_id.get(id).copied()
    }

    /// Returns a reference to the task with the given ID.
    pub fn task_by_id(&self, id: &str) -> Option<&T> {
        self.node_by_id
            .get(id)
            .and_then(|&n| self.graph.node_weight(n))
    }

    /// Returns a mutable reference to the task with the given ID.
    pub fn task_by_id_mut(&mut self, id: &str) -> Option<&mut T> {
        if let Some(&n) = self.node_by_id.get(id) {
            self.graph.node_weight_mut(n)
        } else {
            None
        }
    }

    /// Returns an iterator over `(Id, &Task)` pairs.
    pub fn tasks(&self) -> impl Iterator<Item = (&str, &T)> {
        self.id_by_node
            .iter()
            .filter_map(move |(node, id)| self.graph.node_weight(*node).map(|t| (id.as_str(), t)))
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

    /// Removes the task with the given `id` together with all its incident edges.
    ///
    /// Both ID-map entries are cleaned up atomically so no stale mappings remain.
    /// Returns the removed task, or `None` if `id` is not registered.
    pub fn remove_task(&mut self, id: &str) -> Option<T> {
        let node = self.node_by_id.remove(id)?;
        self.id_by_node.remove(&node);
        self.graph.remove_node(node)
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

        // Use NodeIndex-keyed maps so sparse indices (after removals) are safe.
        let mut earliest_start: HashMap<petgraph::graph::NodeIndex, f64> =
            topo.iter().map(|&n| (n, 0.0)).collect();
        let mut predecessor: HashMap<
            petgraph::graph::NodeIndex,
            Option<petgraph::graph::NodeIndex>,
        > = topo.iter().map(|&n| (n, None)).collect();

        for &node in &topo {
            // Use size_on_axis() for scheduling math
            let task_duration = self.graph[node].size_on_axis().value();
            let node_start = earliest_start[&node];

            for successor in self.successors(node) {
                let new_start = node_start + task_duration;
                let succ_start = earliest_start.entry(successor).or_insert(0.0);

                if new_start > *succ_start {
                    *succ_start = new_start;
                    predecessor.insert(successor, Some(node));
                }
            }
        }

        let mut max_finish = 0.0_f64;
        let mut end_node = None;

        for node in self.graph.node_indices() {
            // Use size_on_axis() for scheduling math
            let finish_time = earliest_start[&node] + self.graph[node].size_on_axis().value();

            if finish_time > max_finish {
                max_finish = finish_time;
                end_node = Some(node);
            }
        }

        let mut path = Vec::new();
        let mut current = end_node;

        while let Some(node) = current {
            path.push(node);
            current = predecessor[&node];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTask;

    // ── Construction ──────────────────────────────────────────────────

    #[test]
    fn new_creates_empty_block() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert_eq!(block.task_count(), 0);
        assert_eq!(block.dependency_count(), 0);
    }

    #[test]
    fn default_creates_empty_block() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::default();
        assert_eq!(block.task_count(), 0);
        assert_eq!(block.dependency_count(), 0);
    }

    // ── Task addition ─────────────────────────────────────────────────

    #[test]
    fn add_task_returns_unique_ids() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let id1 = block.add_task(TestTask::new("A", 10.0));
        let id2 = block.add_task(TestTask::new("B", 20.0));
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
        assert!(!id2.is_empty());
        assert_eq!(block.task_count(), 2);
    }

    #[test]
    fn add_task_with_custom_id() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let id = block
            .add_task_with_id(TestTask::new("A", 10.0), Some("my-custom-id".into()))
            .unwrap();
        assert_eq!(id, "my-custom-id");
    }

    #[test]
    fn add_task_with_none_id_generates_id() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let id = block
            .add_task_with_id(TestTask::new("A", 10.0), None)
            .unwrap();
        assert!(!id.is_empty());
    }

    // ── Lookups ───────────────────────────────────────────────────────

    #[test]
    fn id_of_returns_correct_id() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let _id = block
            .add_task_with_id(TestTask::new("A", 10.0), Some("task-a".into()))
            .unwrap();
        let node = block.node_of("task-a").unwrap();
        assert_eq!(block.id_of(node), Some("task-a"));
    }

    #[test]
    fn id_of_returns_none_for_invalid_index() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert_eq!(block.id_of(petgraph::graph::NodeIndex::new(999)), None);
    }

    #[test]
    fn node_of_returns_correct_index() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let _id = block
            .add_task_with_id(TestTask::new("A", 10.0), Some("task-a".into()))
            .unwrap();
        let node = block.node_of("task-a");
        assert!(node.is_some());
    }

    #[test]
    fn node_of_returns_none_for_unknown_id() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert_eq!(block.node_of("nonexistent"), None);
    }

    #[test]
    fn task_by_id_returns_correct_task() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("Alpha", 10.0), Some("alpha".into()))
            .unwrap();
        let task = block.task_by_id("alpha").unwrap();
        assert_eq!(task.name(), "Alpha");
    }

    #[test]
    fn task_by_id_returns_none_for_unknown() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert!(block.task_by_id("nope").is_none());
    }

    #[test]
    fn task_by_id_mut_modifies_task() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("Before", 10.0), Some("t".into()))
            .unwrap();
        if let Some(task) = block.task_by_id_mut("t") {
            task.name = "After".to_string();
        }
        assert_eq!(block.task_by_id("t").unwrap().name(), "After");
    }

    #[test]
    fn task_by_id_mut_returns_none_for_unknown() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert!(block.task_by_id_mut("nope").is_none());
    }

    #[test]
    fn get_task_by_node() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("X", 5.0), Some("x".into()))
            .unwrap();
        let node = block.node_of("x").unwrap();
        assert_eq!(block.get_task(node).unwrap().name(), "X");
    }

    #[test]
    fn get_task_mut_by_node() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("X", 5.0), Some("x".into()))
            .unwrap();
        let node = block.node_of("x").unwrap();
        block.get_task_mut(node).unwrap().name = "Y".to_string();
        assert_eq!(block.get_task(node).unwrap().name(), "Y");
    }

    #[test]
    fn get_task_returns_none_for_invalid_node() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert!(block
            .get_task(petgraph::graph::NodeIndex::new(999))
            .is_none());
    }

    // ── Tasks iterator ────────────────────────────────────────────────

    #[test]
    fn tasks_yields_all_pairs() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 20.0), Some("b".into()))
            .unwrap();

        let mut pairs: Vec<_> = block
            .tasks()
            .map(|(id, t)| (id.to_owned(), t.name().to_owned()))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("a".to_string(), "A".to_string()),
                ("b".to_string(), "B".to_string())
            ]
        );
    }

    // ── Dependencies ──────────────────────────────────────────────────

    #[test]
    fn add_dependency_ok() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();

        assert!(block.add_dependency(na, nb, ()).is_ok());
        assert_eq!(block.dependency_count(), 1);
    }

    #[test]
    fn add_dependency_cycle_detected() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        let result = block.add_dependency(nb, na, ());
        assert_eq!(result, Err(SchedulingError::CycleDetected));
    }

    #[test]
    fn add_dependency_invalid_from_node() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let bad = petgraph::graph::NodeIndex::new(999);

        let result = block.add_dependency(bad, na, ());
        assert!(matches!(result, Err(SchedulingError::InvalidNodeIndex(_))));
    }

    #[test]
    fn add_dependency_invalid_to_node() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let bad = petgraph::graph::NodeIndex::new(999);

        let result = block.add_dependency(na, bad, ());
        assert!(matches!(result, Err(SchedulingError::InvalidNodeIndex(_))));
    }

    #[test]
    fn add_dependency_self_loop_is_cycle() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();

        let result = block.add_dependency(na, na, ());
        assert_eq!(result, Err(SchedulingError::CycleDetected));
    }

    // ── Topological order ─────────────────────────────────────────────

    #[test]
    fn topo_order_linear_dag() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 10.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(nb, nc, ()).unwrap();

        let order = block.topo_order().unwrap();
        let pos_a = order.iter().position(|n| *n == na).unwrap();
        let pos_b = order.iter().position(|n| *n == nb).unwrap();
        let pos_c = order.iter().position(|n| *n == nc).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn topo_order_empty_block() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let order = block.topo_order().unwrap();
        assert!(order.is_empty());
    }

    // ── Roots and leaves ──────────────────────────────────────────────

    #[test]
    fn roots_and_leaves_linear_chain() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 10.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(nb, nc, ()).unwrap();

        assert_eq!(block.roots(), vec![na]);
        assert_eq!(block.leaves(), vec![nc]);
    }

    #[test]
    fn single_task_is_both_root_and_leaf() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();

        assert_eq!(block.roots(), vec![na]);
        assert_eq!(block.leaves(), vec![na]);
    }

    #[test]
    fn disconnected_tasks_multiple_roots_and_leaves() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();

        assert_eq!(block.roots().len(), 2);
        assert_eq!(block.leaves().len(), 2);
    }

    // ── Predecessors and successors ───────────────────────────────────

    #[test]
    fn predecessors_and_successors() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 10.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(na, nc, ()).unwrap();

        assert!(block.predecessors(na).is_empty());
        assert_eq!(block.predecessors(nb), vec![na]);
        assert_eq!(block.predecessors(nc), vec![na]);

        let succs: Vec<_> = block.successors(na);
        assert_eq!(succs.len(), 2);
        assert!(succs.contains(&nb));
        assert!(succs.contains(&nc));
        assert!(block.successors(nb).is_empty());
    }

    // ── Critical path ─────────────────────────────────────────────────

    #[test]
    fn critical_path_empty_block() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert_eq!(block.critical_path(), Err(SchedulingError::EmptyGraph));
    }

    #[test]
    fn critical_path_single_task() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();

        let (cost, path) = block.critical_path().unwrap();
        assert!((cost - 10.0).abs() < 1e-9);
        assert_eq!(path, vec![na]);
    }

    #[test]
    fn critical_path_linear_chain() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 20.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 30.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(nb, nc, ()).unwrap();

        let (cost, path) = block.critical_path().unwrap();
        assert!((cost - 60.0).abs() < 1e-9);
        assert_eq!(path, vec![na, nb, nc]);
    }

    #[test]
    fn critical_path_diamond_dag() {
        // Diamond: A -> B, A -> C, B -> D, C -> D
        // A=10, B=30, C=5, D=10
        // Path A->B->D = 50, Path A->C->D = 25 → critical = A->B->D
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 30.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 5.0), Some("c".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("D", 10.0), Some("d".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();
        let nd = block.node_of("d").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(na, nc, ()).unwrap();
        block.add_dependency(nb, nd, ()).unwrap();
        block.add_dependency(nc, nd, ()).unwrap();

        let (cost, path) = block.critical_path().unwrap();
        assert!((cost - 50.0).abs() < 1e-9);
        assert_eq!(path, vec![na, nb, nd]);
    }

    // ── Counts ────────────────────────────────────────────────────────

    #[test]
    fn task_count_and_dependency_count() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert_eq!(block.task_count(), 0);
        assert_eq!(block.dependency_count(), 0);

        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        assert_eq!(block.task_count(), 2);

        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        block.add_dependency(na, nb, ()).unwrap();
        assert_eq!(block.dependency_count(), 1);
    }

    // ── Graph access ──────────────────────────────────────────────────

    #[test]
    fn graph_ref_access() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block.add_task(TestTask::new("A", 10.0));
        assert_eq!(block.graph().node_count(), 1);
    }

    #[test]
    fn remove_task_cleans_id_maps() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();

        let removed = block.remove_task("a");
        assert!(removed.is_some());
        assert_eq!(block.task_count(), 1);
        // Both directions of the ID map must be cleaned up.
        assert!(block.task_by_id("a").is_none());
        assert!(block.node_of("a").is_none());
        // The remaining task must still be reachable.
        assert!(block.task_by_id("b").is_some());
    }

    #[test]
    fn remove_task_returns_none_for_unknown_id() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        assert!(block.remove_task("ghost").is_none());
    }

    #[test]
    fn remove_task_strips_incident_edges() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        block.add_dependency(na, nb, ()).unwrap();
        assert_eq!(block.dependency_count(), 1);

        block.remove_task("a");
        // The edge from "a" must have been removed with the node.
        assert_eq!(block.dependency_count(), 0);
    }

    #[test]
    fn critical_path_still_correct_after_removal() {
        // A(10) -> B(20) -> C(30), then remove B; leaves disconnected A(10) and C(30).
        // Critical path is just C with duration 30.
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 20.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 30.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();
        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(nb, nc, ()).unwrap();

        block.remove_task("b");

        // After removing B the edges are also gone, so A and C are disconnected.
        let (cost, path) = block.critical_path().unwrap();
        assert!((cost - 30.0).abs() < 1e-9, "expected 30, got {cost}");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], nc);
    }

    // ── Display ───────────────────────────────────────────────────────

    #[test]
    fn display_contains_task_info() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block.add_task(TestTask::new("Alpha", 10.0));
        block.add_task(TestTask::new("Beta", 20.0));

        let output = format!("{}", block);
        assert!(output.contains("SchedulingBlock"));
        assert!(output.contains("Tasks: 2"));
        assert!(output.contains("Dependencies: 0"));
        assert!(output.contains("Alpha"));
        assert!(output.contains("Beta"));
    }

    #[test]
    fn display_empty_block() {
        let block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        let output = format!("{}", block);
        assert!(output.contains("Tasks: 0"));
    }

    // ── ID uniqueness ─────────────────────────────────────────────────

    #[test]
    fn add_task_with_duplicate_id_returns_error() {
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("shared-id".into()))
            .unwrap();
        let result = block.add_task_with_id(TestTask::new("B", 20.0), Some("shared-id".into()));
        assert_eq!(
            result,
            Err(SchedulingError::DuplicateId("shared-id".to_string()))
        );
        // The original task must still be retrievable and the graph intact
        assert_eq!(block.task_count(), 1);
        assert_eq!(block.task_by_id("shared-id").unwrap().name(), "A");
    }

    // ── Transitive cycle detection ────────────────────────────────────

    #[test]
    fn transitive_cycle_detected() {
        // A -> B -> C, then C -> A should fail
        let mut block: SchedulingBlock<TestTask> = SchedulingBlock::new();
        block
            .add_task_with_id(TestTask::new("A", 10.0), Some("a".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("B", 10.0), Some("b".into()))
            .unwrap();
        block
            .add_task_with_id(TestTask::new("C", 10.0), Some("c".into()))
            .unwrap();
        let na = block.node_of("a").unwrap();
        let nb = block.node_of("b").unwrap();
        let nc = block.node_of("c").unwrap();

        block.add_dependency(na, nb, ()).unwrap();
        block.add_dependency(nb, nc, ()).unwrap();
        let result = block.add_dependency(nc, na, ());
        assert_eq!(result, Err(SchedulingError::CycleDetected));
    }
}
