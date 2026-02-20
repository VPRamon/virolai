//! Algorithm-agnostic evaluation of dynamic constraints.
//!
//! [`DynamicConstraintIndex`] pre-indexes all dynamic constraint edges from one
//! or more [`SchedulingBlock`]s into a lookup table keyed by **target** task ID.
//! Any scheduling algorithm can then evaluate all incoming dynamic constraints
//! for a candidate task with a single call, without traversing the block graph
//! at runtime.
//!
//! # Architecture
//!
//! ```text
//!   SchedulingBlock(s)          DynamicConstraintIndex
//!   ┌──────────────┐           ┌──────────────────────┐
//!   │  A ──D──▶ B  │  build()  │ "B" → [("A", &D)]   │
//!   │  A ──D──▶ C  │ ───────▶  │ "C" → [("A", &D)]   │
//!   │  B ──D──▶ C  │           │ "C" → [("B", &D)]   │
//!   └──────────────┘           └──────────────────────┘
//! ```
//!
//! At each scheduling iteration the algorithm calls
//! [`evaluate()`](DynamicConstraintIndex::evaluate) or
//! [`compute_effective_intervals()`](DynamicConstraintIndex::compute_effective_intervals)
//! to obtain the combined valid intervals for a task after all dynamic
//! constraints are applied.

use super::constraint::{DynamicConstraint, SchedulingContext};
use crate::scheduling_block::{SchedulingBlock, Task};
use crate::solution_space::{Interval, IntervalSet};
use crate::Id;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use qtty::Unit;
use std::collections::HashMap;

/// Pre-built index mapping target task IDs to their incoming dynamic constraints.
///
/// Each entry is `(source_task_id, &constraint)` — the source ID is resolved
/// from the graph edge at build time so that constraints remain stateless w.r.t.
/// task identity.
///
/// # Lifetime
///
/// Borrows edge data from the blocks, so the index lives as long as the blocks.
#[derive(Debug)]
pub struct DynamicConstraintIndex<'a, D> {
    /// `target_task_id → Vec<(source_task_id, &constraint)>`
    edges: HashMap<Id, Vec<(Id, &'a D)>>,
}

impl<'a, D> DynamicConstraintIndex<'a, D> {
    /// Builds an index from one or more scheduling blocks.
    ///
    /// Walks every edge in every block and records `(source_id, &edge_data)`
    /// keyed by `target_id`.
    ///
    /// # Complexity
    ///
    /// O(total edges across all blocks).
    pub fn from_blocks<T, U, E>(blocks: &'a [SchedulingBlock<T, U, D, E>]) -> Self
    where
        T: Task<U>,
        U: Unit,
        E: petgraph::EdgeType,
    {
        let mut edges: HashMap<Id, Vec<(Id, &'a D)>> = HashMap::new();

        for block in blocks {
            let graph = block.graph();
            for edge_ref in graph.edge_references() {
                let source_node = edge_ref.source();
                let target_node = edge_ref.target();

                // Resolve node indices → task IDs via the block's ID map.
                if let (Some(source_id), Some(target_id)) =
                    (block.id_of(source_node), block.id_of(target_node))
                {
                    edges
                        .entry(target_id.to_owned())
                        .or_default()
                        .push((source_id.to_owned(), edge_ref.weight()));
                }
            }
        }

        Self { edges }
    }

    /// Returns the number of target tasks that have dynamic constraints.
    pub fn target_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if `task_id` has any incoming dynamic constraint edges.
    pub fn has_constraints(&self, task_id: &str) -> bool {
        self.edges.contains_key(task_id)
    }

    /// Returns the incoming constraint edges for a target task, if any.
    pub fn get_edges(&self, task_id: &str) -> Option<&[(Id, &'a D)]> {
        self.edges.get(task_id).map(|v| v.as_slice())
    }
}

impl<'a, D> DynamicConstraintIndex<'a, D> {
    /// Evaluates all incoming dynamic constraints for `task_id` and returns
    /// their **intersection** (AND-composition).
    ///
    /// Returns `None` if `task_id` has no dynamic constraints — this lets the
    /// caller skip unnecessary intersection with the static solution space.
    ///
    /// # Complexity
    ///
    /// O(k · C) where k = number of incoming edges and C = cost of a single
    /// constraint evaluation (typically O(1) for built-in kinds).
    pub fn evaluate<U>(
        &self,
        task_id: &str,
        range: Interval<U>,
        ctx: &SchedulingContext<U>,
    ) -> Option<IntervalSet<U>>
    where
        D: DynamicConstraint<U>,
        U: Unit,
    {
        let incoming = self.edges.get(task_id)?;
        if incoming.is_empty() {
            return None;
        }

        let result = incoming
            .iter()
            .map(|(source_id, constraint)| constraint.compute_intervals(range, source_id, ctx))
            .reduce(|acc, v| crate::constraints::operations::compute_intersection(&acc, &v))
            .unwrap_or_default();

        Some(result)
    }

    /// Computes the **effective** intervals for a task by intersecting the
    /// static solution space intervals with the dynamic constraint overlay.
    ///
    /// If the task has no dynamic constraints, returns the static intervals
    /// unchanged (zero-cost path).
    ///
    /// # Arguments
    ///
    /// * `task_id` — the candidate task
    /// * `static_intervals` — pre-computed intervals from static constraints
    /// * `range` — the current scheduling horizon / query range
    /// * `ctx` — current scheduling context (partial schedule + solution space)
    pub fn compute_effective_intervals<U>(
        &self,
        task_id: &str,
        static_intervals: &IntervalSet<U>,
        range: Interval<U>,
        ctx: &SchedulingContext<U>,
    ) -> IntervalSet<U>
    where
        D: DynamicConstraint<U>,
        U: Unit,
    {
        match self.evaluate(task_id, range, ctx) {
            Some(dynamic_intervals) => crate::constraints::operations::compute_intersection(
                static_intervals,
                &dynamic_intervals,
            ),
            None => static_intervals.clone(),
        }
    }
}

impl<'a, D> Default for DynamicConstraintIndex<'a, D> {
    fn default() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::hard::dynamic::DynConstraintKind;
    use crate::schedule::Schedule;
    use crate::solution_space::{Interval, SolutionSpace};
    use crate::test_utils::TestTask;
    use qtty::Second;

    fn iv(start: f64, end: f64) -> Interval<Second> {
        Interval::from_f64(start, end)
    }

    // ── from_blocks ───────────────────────────────────────────────────

    #[test]
    fn empty_blocks_produce_empty_index() {
        let blocks: Vec<SchedulingBlock<TestTask, Second, DynConstraintKind>> = vec![];
        let index = DynamicConstraintIndex::from_blocks(&blocks);
        assert_eq!(index.target_count(), 0);
    }

    #[test]
    fn block_without_edges_produces_empty_index() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        block.add_task(TestTask::new("A", 10.0));
        block.add_task(TestTask::new("B", 10.0));

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);
        assert_eq!(index.target_count(), 0);
    }

    #[test]
    fn single_edge_indexes_correctly() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_b = block.add_task(TestTask::new("B", 10.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        block
            .add_dependency(node_a, node_b, DynConstraintKind::Consecutive)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);
        assert_eq!(index.target_count(), 1);
        assert!(index.has_constraints(&id_b));
        assert!(!index.has_constraints(&id_a));

        let edges = index.get_edges(&id_b).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, id_a);
        assert_eq!(*edges[0].1, DynConstraintKind::Consecutive);
    }

    // ── evaluate ──────────────────────────────────────────────────────

    #[test]
    fn evaluate_no_constraints_returns_none() {
        let blocks: Vec<SchedulingBlock<TestTask, Second, DynConstraintKind>> = vec![];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        let schedule = Schedule::new();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        assert!(index
            .evaluate("nonexistent", iv(0.0, 100.0), &ctx)
            .is_none());
    }

    #[test]
    fn evaluate_dependence_ref_scheduled() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_b = block.add_task(TestTask::new("B", 10.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        block
            .add_dependency(node_a, node_b, DynConstraintKind::Dependence)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        let mut schedule = Schedule::new();
        schedule.add(&id_a, iv(0.0, 10.0)).unwrap();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        let result = index.evaluate(&id_b, iv(0.0, 100.0), &ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(0.0, 100.0));
    }

    #[test]
    fn evaluate_dependence_ref_absent() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_b = block.add_task(TestTask::new("B", 10.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        block
            .add_dependency(node_a, node_b, DynConstraintKind::Dependence)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        let schedule = Schedule::new();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        let result = index.evaluate(&id_b, iv(0.0, 100.0), &ctx).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn evaluate_consecutive_returns_after_ref_end() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_b = block.add_task(TestTask::new("B", 15.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        block
            .add_dependency(node_a, node_b, DynConstraintKind::Consecutive)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        let mut schedule = Schedule::new();
        schedule.add(&id_a, iv(5.0, 15.0)).unwrap();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        let result = index.evaluate(&id_b, iv(0.0, 100.0), &ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(15.0, 100.0));
    }

    #[test]
    fn evaluate_multiple_edges_intersected() {
        // B depends on A (Dependence) AND must come after C (Consecutive)
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_c = block.add_task(TestTask::new("C", 20.0));
        let id_b = block.add_task(TestTask::new("B", 10.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        let node_c = block.node_of(&id_c).unwrap();

        block
            .add_dependency(node_a, node_b, DynConstraintKind::Dependence)
            .unwrap();
        block
            .add_dependency(node_c, node_b, DynConstraintKind::Consecutive)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        // Schedule A at [0,10) and C at [10,30)
        let mut schedule = Schedule::new();
        schedule.add(&id_a, iv(0.0, 10.0)).unwrap();
        schedule.add(&id_c, iv(10.0, 30.0)).unwrap();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        // Dependence(A) → full range, Consecutive(C) → [30, 100)
        // Intersection → [30, 100)
        let result = index.evaluate(&id_b, iv(0.0, 100.0), &ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], iv(30.0, 100.0));
    }

    // ── compute_effective_intervals ───────────────────────────────────

    #[test]
    fn effective_intervals_no_dynamic_returns_static() {
        let blocks: Vec<SchedulingBlock<TestTask, Second, DynConstraintKind>> = vec![];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        let schedule = Schedule::new();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        let static_ivs = IntervalSet::from(vec![iv(10.0, 50.0), iv(60.0, 90.0)]);
        let result = index.compute_effective_intervals("task-x", &static_ivs, iv(0.0, 100.0), &ctx);
        assert_eq!(result, static_ivs);
    }

    #[test]
    fn effective_intervals_with_dynamic_overlay() {
        let mut block: SchedulingBlock<TestTask, Second, DynConstraintKind> =
            SchedulingBlock::new();
        let id_a = block.add_task(TestTask::new("A", 10.0));
        let id_b = block.add_task(TestTask::new("B", 10.0));
        let node_a = block.node_of(&id_a).unwrap();
        let node_b = block.node_of(&id_b).unwrap();
        block
            .add_dependency(node_a, node_b, DynConstraintKind::Consecutive)
            .unwrap();

        let blocks = vec![block];
        let index = DynamicConstraintIndex::from_blocks(&blocks);

        // A scheduled at [0, 10) → B valid from [10, 100)
        let mut schedule = Schedule::new();
        schedule.add(&id_a, iv(0.0, 10.0)).unwrap();
        let ss = SolutionSpace::new();
        let ctx = SchedulingContext::new(&schedule, &ss);

        // Static says B is valid in [5, 50) and [60, 90)
        let static_ivs = IntervalSet::from(vec![iv(5.0, 50.0), iv(60.0, 90.0)]);
        // Dynamic overlay: [10, 100)
        // Intersection: [10, 50) and [60, 90)
        let result = index.compute_effective_intervals(&id_b, &static_ivs, iv(0.0, 100.0), &ctx);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], iv(10.0, 50.0));
        assert_eq!(result[1], iv(60.0, 90.0));
    }
}
