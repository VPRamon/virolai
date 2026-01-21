# Constraints Module

Composable constraint trees for producing valid scheduling intervals (`Interval<U>`) over a query range.

At a glance:
- Leaf constraints implement [`Constraint`](crate::constraints::Constraint) and return a canonical interval set (sorted, non-overlapping).
- [`ConstraintNode`](crate::constraints::ConstraintNode) composes leaves using boolean logic: `Intersection` (AND), `Union` (OR), and `Not` (complement within the query range).

```
            Union (OR)
           /          \
    Intersection      Leaf C
      (AND)
     /     \
  Leaf A   Leaf B
```

## Core API

### `Constraint`

The `Constraint` trait is the interface for all leaf constraints:

```rust
use virolai::scheduling_block::{Interval, SolutionSpace};
use qtty::Unit;
use std::fmt::Debug;

pub trait Constraint<U: Unit>: Send + Sync + Debug {
    fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>>;
    fn stringify(&self) -> String;

    fn populate_solution_space(&self, solution_space: &mut SolutionSpace<U>, range: Interval<U>);
}
```

Contract for `compute_intervals` (relied on by set-operations):
- Returns intervals within the provided `range`.
- Output should be canonical: sorted by start, non-overlapping.

### `ConstraintNode`

`ConstraintNode` is the composable tree type:

```rust
use qtty::Unit;
use std::sync::Arc;
use virolai::constraints::Constraint;

pub enum ConstraintNode<U: Unit> {
    Leaf(Arc<dyn Constraint<U>>),
    Not(Arc<ConstraintNode<U>>),
    Intersection(Vec<ConstraintNode<U>>),
    Union(Vec<ConstraintNode<U>>),
}
```

Construction helpers:
- `ConstraintNode::leaf(constraint)`
- `ConstraintNode::not(child)`
- `ConstraintNode::intersection(children)`
- `ConstraintNode::union(children)`

Evaluation:
- `ConstraintNode::compute_intervals(range)` recursively evaluates the tree and returns a canonical interval set.

`ConstraintNode<U>` also implements `Constraint<U>`, so you can pass a whole tree anywhere a `Constraint` is expected.

### `IntervalConstraint`

`IntervalConstraint` is a simple fixed-window leaf constraint:

```rust
use qtty::Second;
use virolai::constraints::IntervalConstraint;
use virolai::scheduling_block::Interval;

let window = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 100.0));
```

## Usage examples

### Example 1: Simple intersection (AND)

```rust
use qtty::Second;
use virolai::constraints::{ConstraintNode, IntervalConstraint};
use virolai::scheduling_block::Interval;

let a = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 100.0));
let b = IntervalConstraint::new(Interval::<Second>::from_f64(50.0, 150.0));

let tree = ConstraintNode::intersection(vec![ConstraintNode::leaf(a), ConstraintNode::leaf(b)]);

let valid = tree.compute_intervals(Interval::<Second>::from_f64(0.0, 200.0));
// => [[50, 100]]
```

### Example 2: Union (OR)

```rust
use qtty::Second;
use virolai::constraints::{ConstraintNode, IntervalConstraint};
use virolai::scheduling_block::Interval;

let morning = IntervalConstraint::new(Interval::<Second>::from_f64(6.0 * 3600.0, 12.0 * 3600.0));
let evening = IntervalConstraint::new(Interval::<Second>::from_f64(18.0 * 3600.0, 23.0 * 3600.0));

let tree = ConstraintNode::union(vec![ConstraintNode::leaf(morning), ConstraintNode::leaf(evening)]);
let valid = tree.compute_intervals(Interval::<Second>::from_f64(0.0, 24.0 * 3600.0));
```

### Example 3: Exclusion with NOT

`Not(child)` computes the complement of `child` within the query range.

```rust
use qtty::Second;
use virolai::constraints::{ConstraintNode, IntervalConstraint};
use virolai::scheduling_block::Interval;

let allowed = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 100.0)));
let forbidden = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(20.0, 40.0)));

let tree = ConstraintNode::intersection(vec![allowed, ConstraintNode::not(forbidden)]);
let valid = tree.compute_intervals(Interval::<Second>::from_f64(0.0, 100.0));
// => [[0, 20], [40, 100]]
```

### Example 4: Dynamic tree building

```rust
use qtty::Second;
use virolai::constraints::{ConstraintNode, IntervalConstraint};
use virolai::scheduling_block::Interval;

let mut tree = ConstraintNode::intersection(Vec::new());
for i in 0..5 {
    let window = IntervalConstraint::new(Interval::<Second>::from_f64(i as f64 * 10.0, 100.0));
    tree.add_child(ConstraintNode::leaf(window)).unwrap();
}

let valid = tree.compute_intervals(Interval::<Second>::from_f64(0.0, 100.0));
```

## Tree analysis and traversal

These helpers live in `src/constraints/tree.rs` and are implemented as methods on `ConstraintNode`:

```rust
let depth = tree.depth();
let nodes = tree.node_count();
let leaves = tree.leaf_count();

tree.visit_preorder(&mut |node| println!("{}", node.stringify()));
tree.visit_leaves(&mut |leaf| println!("{}", leaf.stringify()));
```

### Flattening

`flatten()` returns a new tree with nested intersections/unions collapsed:

```rust
let flattened = tree.flatten();
```

## Interval set operations

`ConstraintNode` uses interval set operations from `src/constraints/operations/`:
- `compute_intersection(a, b)`
- `compute_union(a, b)`
- `compute_complement(canonical, range)`

In debug builds, `compute_intersection`/`compute_union` include `debug_assert!` checks that inputs are canonical.

## Module structure

```
src/constraints/
├── mod.rs              # Module exports + helpers (quantity_min/quantity_max)
├── constraint.rs       # Constraint trait + IntervalConstraint
├── node.rs             # ConstraintNode + evaluation
├── tree.rs             # Tree utilities (depth/traversal/flatten/add_child)
├── error.rs            # ConstraintError for tree editing
└── operations/         # Interval set operations (union/intersection/complement)
```
