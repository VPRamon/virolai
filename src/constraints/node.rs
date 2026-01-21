//! Composable constraint trees with AND/OR logic.
//!
//! The core type [`ConstraintExpr<C>`] is generic over the leaf constraint type,
//! enabling serde serialization when the leaf type is serializable.

use crate::constraints::constraint::Constraint;
use crate::solution_space::Interval;
use qtty::Unit;
use std::ops::Not;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Constraint tree node: leaf (concrete constraint) or combinator (AND/OR/NOT).
///
/// Trees enable composing complex scheduling logic:
/// - **Leaf**: Wraps a concrete constraint of type `C`
/// - **Not**: Logical negation of a subtree
/// - **Intersection**: AND logic – all children must be satisfied
/// - **Union**: OR logic – at least one child must be satisfied
///
/// # Serde Support
///
/// When the `serde` feature is enabled and the leaf type `C` implements
/// `Serialize`/`Deserialize`, the entire tree is automatically serializable.
///
/// # Example
///
/// ```ignore
/// use vrolai::constraints::ConstraintExpr;
///
/// // Define your leaf enum
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// #[serde(tag = "type")]
/// enum MyLeaf {
///     TimeWindow { start: f64, end: f64 },
///     Custom(CustomConstraint),
/// }
///
/// // Build a tree
/// let tree: ConstraintExpr<MyLeaf> = ConstraintExpr::intersection(vec![
///     ConstraintExpr::leaf(MyLeaf::TimeWindow { start: 0.0, end: 100.0 }),
///     ConstraintExpr::not(ConstraintExpr::leaf(MyLeaf::Custom(c))),
/// ]);
///
/// // Serialize/deserialize works automatically
/// let json = serde_json::to_string(&tree)?;
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "op", rename_all = "snake_case"))]
pub enum ConstraintExpr<C> {
    /// A leaf constraint.
    Leaf(C),
    /// Logical NOT of a subtree.
    Not {
        /// The child subtree to negate.
        child: Box<ConstraintExpr<C>>,
    },
    /// Logical AND (intersection) of multiple subtrees.
    Intersection {
        /// Child subtrees (all must be satisfied).
        children: Vec<ConstraintExpr<C>>,
    },
    /// Logical OR (union) of multiple subtrees.
    Union {
        /// Child subtrees (at least one must be satisfied).
        children: Vec<ConstraintExpr<C>>,
    },
}

impl<C> ConstraintExpr<C> {
    /// Creates a leaf node wrapping a constraint.
    pub fn leaf(constraint: C) -> Self {
        ConstraintExpr::Leaf(constraint)
    }

    /// Creates a NOT node wrapping a subtree.
    pub fn not(child: ConstraintExpr<C>) -> Self {
        ConstraintExpr::Not {
            child: Box::new(child),
        }
    }

    /// Creates an intersection node (AND logic).
    pub fn intersection(children: Vec<ConstraintExpr<C>>) -> Self {
        ConstraintExpr::Intersection { children }
    }

    /// Creates a union node (OR logic).
    pub fn union(children: Vec<ConstraintExpr<C>>) -> Self {
        ConstraintExpr::Union { children }
    }

    /// Returns whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, ConstraintExpr::Leaf(_))
    }

    /// Returns whether this node is a negation (NOT).
    pub fn is_not(&self) -> bool {
        matches!(self, ConstraintExpr::Not { .. })
    }

    /// Returns whether this node is an intersection.
    pub fn is_intersection(&self) -> bool {
        matches!(self, ConstraintExpr::Intersection { .. })
    }

    /// Returns whether this node is a union.
    pub fn is_union(&self) -> bool {
        matches!(self, ConstraintExpr::Union { .. })
    }

    /// Returns the depth of this constraint tree.
    ///
    /// - Leaf nodes have depth 1
    /// - Interior nodes have depth = 1 + max(child depths)
    pub fn depth(&self) -> usize {
        match self {
            ConstraintExpr::Leaf(_) => 1,
            ConstraintExpr::Not { child } => 1 + child.depth(),
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Returns the total number of nodes in this constraint tree.
    pub fn node_count(&self) -> usize {
        match self {
            ConstraintExpr::Leaf(_) => 1,
            ConstraintExpr::Not { child } => 1 + child.node_count(),
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Returns the number of leaf nodes in this constraint tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            ConstraintExpr::Leaf(_) => 1,
            ConstraintExpr::Not { child } => child.leaf_count(),
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                children.iter().map(|c| c.leaf_count()).sum()
            }
        }
    }

    /// Returns a reference to the children if this is a combinator node (Intersection/Union).
    pub fn children(&self) -> Option<&[ConstraintExpr<C>]> {
        match self {
            ConstraintExpr::Leaf(_) | ConstraintExpr::Not { .. } => None,
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                Some(children)
            }
        }
    }

    /// Returns a mutable reference to the children if this is a combinator node.
    pub fn children_mut(&mut self) -> Option<&mut Vec<ConstraintExpr<C>>> {
        match self {
            ConstraintExpr::Leaf(_) | ConstraintExpr::Not { .. } => None,
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                Some(children)
            }
        }
    }

    /// Visits all nodes in the tree in pre-order (depth-first).
    pub fn visit_preorder<F>(&self, visitor: &mut F)
    where
        F: FnMut(&ConstraintExpr<C>),
    {
        visitor(self);
        match self {
            ConstraintExpr::Leaf(_) => {}
            ConstraintExpr::Not { child } => child.visit_preorder(visitor),
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                for child in children {
                    child.visit_preorder(visitor);
                }
            }
        }
    }

    /// Visits all leaf constraints in the tree.
    pub fn visit_leaves<F>(&self, visitor: &mut F)
    where
        F: FnMut(&C),
    {
        match self {
            ConstraintExpr::Leaf(constraint) => visitor(constraint),
            ConstraintExpr::Not { child } => child.visit_leaves(visitor),
            ConstraintExpr::Intersection { children } | ConstraintExpr::Union { children } => {
                for child in children {
                    child.visit_leaves(visitor);
                }
            }
        }
    }

    /// Transforms each leaf in the tree using the provided function.
    pub fn map_leaves<F, D>(self, f: &mut F) -> ConstraintExpr<D>
    where
        F: FnMut(C) -> D,
    {
        match self {
            ConstraintExpr::Leaf(c) => ConstraintExpr::Leaf(f(c)),
            ConstraintExpr::Not { child } => ConstraintExpr::Not {
                child: Box::new(child.map_leaves(f)),
            },
            ConstraintExpr::Intersection { children } => ConstraintExpr::Intersection {
                children: children.into_iter().map(|c| c.map_leaves(f)).collect(),
            },
            ConstraintExpr::Union { children } => ConstraintExpr::Union {
                children: children.into_iter().map(|c| c.map_leaves(f)).collect(),
            },
        }
    }

    /// Prints the tree structure in a human-readable format.
    ///
    /// # Arguments
    ///
    /// * `indent` - The current indentation level (use 0 for the root)
    /// * `leaf_fmt` - A function to format leaf constraints
    pub fn print_tree_with<F>(&self, indent: usize, leaf_fmt: &F)
    where
        F: Fn(&C) -> String,
    {
        let prefix = "  ".repeat(indent);
        match self {
            ConstraintExpr::Leaf(constraint) => {
                println!("{}└─ Leaf: {}", prefix, leaf_fmt(constraint));
            }
            ConstraintExpr::Not { child } => {
                println!("{}└─ Not", prefix);
                child.print_tree_with(indent + 1, leaf_fmt);
            }
            ConstraintExpr::Intersection { children } => {
                println!("{}└─ Intersection", prefix);
                for child in children {
                    child.print_tree_with(indent + 1, leaf_fmt);
                }
            }
            ConstraintExpr::Union { children } => {
                println!("{}└─ Union", prefix);
                for child in children {
                    child.print_tree_with(indent + 1, leaf_fmt);
                }
            }
        }
    }
}

impl<C: Clone> ConstraintExpr<C> {
    /// Flattens nested combinators of the same type.
    ///
    /// For example, `Intersection(Intersection(A, B), C)` becomes `Intersection(A, B, C)`.
    /// This can improve performance and tree readability.
    pub fn flatten(&self) -> ConstraintExpr<C> {
        match self {
            ConstraintExpr::Leaf(constraint) => ConstraintExpr::Leaf(constraint.clone()),
            ConstraintExpr::Not { child } => ConstraintExpr::Not {
                child: Box::new(child.flatten()),
            },
            ConstraintExpr::Intersection { children } => {
                let mut new_children = Vec::with_capacity(children.len() * 2);
                for child in children {
                    let flattened = child.flatten();
                    if let ConstraintExpr::Intersection {
                        children: mut nested,
                    } = flattened
                    {
                        new_children.append(&mut nested);
                    } else {
                        new_children.push(flattened);
                    }
                }
                new_children.shrink_to_fit();
                ConstraintExpr::Intersection {
                    children: new_children,
                }
            }
            ConstraintExpr::Union { children } => {
                let mut new_children = Vec::with_capacity(children.len() * 2);
                for child in children {
                    let flattened = child.flatten();
                    if let ConstraintExpr::Union {
                        children: mut nested,
                    } = flattened
                    {
                        new_children.append(&mut nested);
                    } else {
                        new_children.push(flattened);
                    }
                }
                new_children.shrink_to_fit();
                ConstraintExpr::Union {
                    children: new_children,
                }
            }
        }
    }
}

// Implement `Not` operator for ergonomic negation: `!tree`
impl<C> Not for ConstraintExpr<C> {
    type Output = Self;

    fn not(self) -> Self {
        ConstraintExpr::not(self)
    }
}

// Implement the Constraint trait for trees whose leaves are constraints.
// This enables using a tree anywhere a single constraint is expected.
impl<U, C> Constraint<U> for ConstraintExpr<C>
where
    U: Unit,
    C: Constraint<U>,
{
    fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>> {
        match self {
            ConstraintExpr::Leaf(constraint) => constraint.compute_intervals(range),
            ConstraintExpr::Not { child } => {
                super::operations::compute_complement(child.compute_intervals(range), range)
            }
            ConstraintExpr::Intersection { children } => children
                .iter()
                .map(|c| c.compute_intervals(range))
                .reduce(|acc, v| super::operations::compute_intersection(&acc, &v))
                .unwrap_or_default(),
            ConstraintExpr::Union { children } => children
                .iter()
                .map(|c| c.compute_intervals(range))
                .fold(Vec::new(), |acc, v| {
                    super::operations::compute_union(&acc, &v)
                }),
        }
    }

    fn stringify(&self) -> String {
        match self {
            ConstraintExpr::Leaf(constraint) => constraint.stringify(),
            ConstraintExpr::Not { child } => format!("Not({})", child.stringify()),
            ConstraintExpr::Intersection { children } => format!(
                "Intersection({})",
                children
                    .iter()
                    .map(|c| c.stringify())
                    .collect::<Vec<_>>()
                    .join(" ∩ ")
            ),
            ConstraintExpr::Union { children } => format!(
                "Union({})",
                children
                    .iter()
                    .map(|c| c.stringify())
                    .collect::<Vec<_>>()
                    .join(" ∪ ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use crate::solution_space::Interval;
    use qtty::{Quantity, Second};

    #[test]
    fn test_intersection_node() {
        let child1 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                0.0, 100.0,
            )));
        let child2 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                50.0, 150.0,
            )));

        let intersection = ConstraintExpr::intersection(vec![child1, child2]);

        let intervals = intersection.compute_intervals(Interval::new(
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(200.0),
        ));

        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start().value(), 50.0);
        assert_eq!(intervals[0].end().value(), 100.0);
    }

    #[test]
    fn test_union_node() {
        let child1 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                0.0, 50.0,
            )));
        let child2 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                100.0, 150.0,
            )));

        let union = ConstraintExpr::union(vec![child1, child2]);

        let intervals = union.compute_intervals(Interval::new(
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(200.0),
        ));

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 50.0);
        assert_eq!(intervals[1].start().value(), 100.0);
        assert_eq!(intervals[1].end().value(), 150.0);
    }

    #[test]
    fn test_complex_tree() {
        // Create a complex tree: (A ∩ B) ∪ (C ∩ D)
        let a = ConstraintExpr::leaf(IntervalConstraint::new(Interval::from_f64(0.0, 100.0)));
        let b = ConstraintExpr::leaf(IntervalConstraint::new(Interval::from_f64(50.0, 150.0)));
        let c = ConstraintExpr::leaf(IntervalConstraint::new(Interval::from_f64(200.0, 300.0)));
        let d = ConstraintExpr::leaf(IntervalConstraint::new(Interval::from_f64(250.0, 350.0)));

        let intersection1 = ConstraintExpr::intersection(vec![a, b]);
        let intersection2 = ConstraintExpr::intersection(vec![c, d]);
        let union = ConstraintExpr::union(vec![intersection1, intersection2]);

        let intervals = union.compute_intervals(Interval::new(
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(400.0),
        ));

        // (A ∩ B) = [50, 100], (C ∩ D) = [250, 300]
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start().value(), 50.0);
        assert_eq!(intervals[0].end().value(), 100.0);
        assert_eq!(intervals[1].start().value(), 250.0);
        assert_eq!(intervals[1].end().value(), 300.0);
    }

    #[test]
    fn test_not_node_complement() {
        // NOT([20, 40] ∪ [60, 80]) over [0, 100] => [0, 20], [40, 60], [80, 100]
        let forbidden = ConstraintExpr::union(vec![
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                20.0, 40.0,
            ))),
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                60.0, 80.0,
            ))),
        ]);

        let not = ConstraintExpr::not(forbidden);

        let intervals = not.compute_intervals(Interval::new(
            Quantity::<Second>::new(0.0),
            Quantity::<Second>::new(100.0),
        ));

        assert_eq!(intervals.len(), 3);
        assert_eq!(
            intervals[0],
            Interval::new(Quantity::<Second>::new(0.0), Quantity::<Second>::new(20.0))
        );
        assert_eq!(
            intervals[1],
            Interval::new(Quantity::<Second>::new(40.0), Quantity::<Second>::new(60.0))
        );
        assert_eq!(
            intervals[2],
            Interval::new(
                Quantity::<Second>::new(80.0),
                Quantity::<Second>::new(100.0)
            )
        );
    }

    #[test]
    fn test_leaf_node() {
        let constraint = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 100.0));
        let leaf = ConstraintExpr::leaf(constraint);

        assert!(leaf.is_leaf());
        assert_eq!(leaf.depth(), 1);
        assert_eq!(leaf.leaf_count(), 1);
        assert_eq!(leaf.node_count(), 1);
    }

    #[test]
    fn test_tree_structure() {
        let child1 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                0.0, 100.0,
            )));
        let child2 =
            ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                50.0, 150.0,
            )));

        let intersection = ConstraintExpr::intersection(vec![child1, child2]);

        assert!(intersection.is_intersection());
        assert_eq!(intersection.depth(), 2);
        assert_eq!(intersection.leaf_count(), 2);
        assert_eq!(intersection.node_count(), 3);
    }

    #[test]
    fn test_flatten() {
        let a = ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 100.0,
        )));
        let b = ConstraintExpr::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            50.0, 150.0,
        )));

        let nested = ConstraintExpr::intersection(vec![ConstraintExpr::intersection(vec![a, b])]);

        let flattened = nested.flatten();

        assert_eq!(flattened.leaf_count(), 2);
        assert!(flattened.is_intersection());
        assert_eq!(flattened.children().unwrap().len(), 2);
    }

    #[test]
    fn test_map_leaves() {
        let tree: ConstraintExpr<i32> = ConstraintExpr::intersection(vec![
            ConstraintExpr::leaf(1),
            ConstraintExpr::not(ConstraintExpr::leaf(2)),
        ]);

        let doubled = tree.map_leaves(&mut |x| x * 2);

        let mut leaves = Vec::new();
        doubled.visit_leaves(&mut |x| leaves.push(*x));
        assert_eq!(leaves, vec![2, 4]);
    }

    #[test]
    fn test_not_operator() {
        let tree: ConstraintExpr<i32> = ConstraintExpr::leaf(42);
        let negated = !tree;
        assert!(negated.is_not());
    }

    /// Test serde roundtrip for ConstraintExpr with simple leaf type.
    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum TestLeaf {
            TimeWindow { start: f64, end: f64 },
            Altitude { min: f64, max: f64 },
        }

        // Build a complex tree
        let tree: ConstraintExpr<TestLeaf> = ConstraintExpr::intersection(vec![
            ConstraintExpr::leaf(TestLeaf::TimeWindow {
                start: 0.0,
                end: 100.0,
            }),
            ConstraintExpr::union(vec![
                ConstraintExpr::leaf(TestLeaf::Altitude {
                    min: 30.0,
                    max: 70.0,
                }),
                ConstraintExpr::not(ConstraintExpr::leaf(TestLeaf::TimeWindow {
                    start: 50.0,
                    end: 60.0,
                })),
            ]),
        ]);

        // Serialize
        let json = serde_json::to_string(&tree).expect("Failed to serialize");

        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["op"], "intersection");
        assert!(parsed["children"].is_array());

        // Deserialize back
        let restored: ConstraintExpr<TestLeaf> =
            serde_json::from_str(&json).expect("Failed to deserialize");

        // Verify structure is preserved
        assert!(restored.is_intersection());
        assert_eq!(restored.leaf_count(), 3);
        assert_eq!(restored.node_count(), 6); // intersection(1), leaf(1), union(1), leaf(1), not(1), leaf(1) = 6

        // Verify leaves match
        let mut original_leaves = Vec::new();
        tree.visit_leaves(&mut |l| original_leaves.push(l.clone()));
        let mut restored_leaves = Vec::new();
        restored.visit_leaves(&mut |l| restored_leaves.push(l.clone()));
        assert_eq!(original_leaves, restored_leaves);
    }
}
