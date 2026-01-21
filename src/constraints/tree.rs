//! Tree analysis operations: depth, node counts, type checks, and traversal.

use super::error::ConstraintError;
use super::ConstraintNode;
use crate::constraints::constraint::Constraint;
use qtty::Unit;
use std::sync::Arc;

impl<U: Unit> ConstraintNode<U> {
    /// Returns the depth of this constraint tree.
    ///
    /// - Leaf nodes have depth 1
    /// - Interior nodes have depth = 1 + max(child depths)
    pub fn depth(&self) -> usize {
        match self {
            ConstraintNode::Leaf(_) => 1,
            ConstraintNode::Not(child) => 1 + child.depth(),
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Returns the total number of nodes in this constraint tree.
    pub fn node_count(&self) -> usize {
        match self {
            ConstraintNode::Leaf(_) => 1,
            ConstraintNode::Not(child) => 1 + child.node_count(),
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Returns the number of leaf nodes in this constraint tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            ConstraintNode::Leaf(_) => 1,
            ConstraintNode::Not(child) => child.leaf_count(),
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                children.iter().map(|c| c.leaf_count()).sum()
            }
        }
    }

    /// Returns whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, ConstraintNode::Leaf(_))
    }

    /// Returns whether this node is a negation (NOT).
    pub fn is_not(&self) -> bool {
        matches!(self, ConstraintNode::Not(_))
    }

    /// Returns whether this node is an intersection.
    pub fn is_intersection(&self) -> bool {
        matches!(self, ConstraintNode::Intersection(_))
    }

    /// Returns whether this node is a union.
    pub fn is_union(&self) -> bool {
        matches!(self, ConstraintNode::Union(_))
    }

    /// Visits all nodes in the tree in pre-order (depth-first).
    ///
    /// # Arguments
    ///
    /// * `visitor` - A closure called for each node
    pub fn visit_preorder<F>(&self, visitor: &mut F)
    where
        F: FnMut(&ConstraintNode<U>),
    {
        visitor(self);
        match self {
            ConstraintNode::Leaf(_) => {}
            ConstraintNode::Not(child) => child.visit_preorder(visitor),
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                for child in children {
                    child.visit_preorder(visitor);
                }
            }
        }
    }

    /// Visits all leaf nodes in the tree.
    ///
    /// # Arguments
    ///
    /// * `visitor` - A closure called for each leaf node
    pub fn visit_leaves<F>(&self, visitor: &mut F)
    where
        F: FnMut(&std::sync::Arc<dyn Constraint<U>>),
    {
        match self {
            ConstraintNode::Leaf(constraint) => visitor(constraint),
            ConstraintNode::Not(child) => child.visit_leaves(visitor),
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                for child in children {
                    child.visit_leaves(visitor);
                }
            }
        }
    }

    /// Flattens nested combinators of the same type.
    ///
    /// For example, `Intersection(Intersection(A, B), C)` becomes `Intersection(A, B, C)`.
    /// This can improve performance and tree readability.
    ///
    /// Since nodes are shared via Arc, this returns a new flattened tree rather than mutating.
    pub fn flatten(&self) -> ConstraintNode<U> {
        match self {
            ConstraintNode::Leaf(constraint) => ConstraintNode::Leaf(constraint.clone()),
            ConstraintNode::Not(child) => ConstraintNode::Not(Arc::new(child.as_ref().flatten())),
            ConstraintNode::Intersection(children) => {
                let mut new_children = Vec::with_capacity(children.len() * 2);
                for child in children {
                    let flattened = child.flatten();
                    if let ConstraintNode::Intersection(mut nested_children) = flattened {
                        new_children.append(&mut nested_children);
                    } else {
                        new_children.push(flattened);
                    }
                }
                new_children.shrink_to_fit();
                ConstraintNode::Intersection(new_children)
            }
            ConstraintNode::Union(children) => {
                let mut new_children = Vec::with_capacity(children.len() * 2);
                for child in children {
                    let flattened = child.flatten();
                    if let ConstraintNode::Union(mut nested_children) = flattened {
                        new_children.append(&mut nested_children);
                    } else {
                        new_children.push(flattened);
                    }
                }
                new_children.shrink_to_fit();
                ConstraintNode::Union(new_children)
            }
        }
    }

    /// Returns a reference to the children if this is a combinator node.
    pub fn children(&self) -> Option<&[ConstraintNode<U>]> {
        match self {
            ConstraintNode::Leaf(_) => None,
            ConstraintNode::Not(_) => None,
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                Some(children)
            }
        }
    }

    /// Returns a mutable reference to the children if this is a combinator node.
    pub fn children_mut(&mut self) -> Option<&mut Vec<ConstraintNode<U>>> {
        match self {
            ConstraintNode::Leaf(_) => None,
            ConstraintNode::Not(_) => None,
            ConstraintNode::Intersection(children) | ConstraintNode::Union(children) => {
                Some(children)
            }
        }
    }

    /// Adds a child constraint to this node if it's a combinator (Intersection or Union).
    ///
    /// Returns `Ok(())` if the child was added, or `Err(ConstraintError)` if this is a leaf or NOT node.
    ///
    /// # Arguments
    ///
    /// * `child` - The child constraint node to add
    pub fn add_child(&mut self, child: ConstraintNode<U>) -> Result<(), ConstraintError> {
        match self {
            ConstraintNode::Leaf(_) => Err(ConstraintError::CannotAddChildToLeaf),
            ConstraintNode::Not(_) => Err(ConstraintError::CannotAddChildToNot),
            ConstraintNode::Intersection(children) => {
                children.push(child);
                Ok(())
            }
            ConstraintNode::Union(children) => {
                children.push(child);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use crate::solution_space::Interval;
    use qtty::Second;

    #[test]
    fn test_leaf_node() {
        let constraint = IntervalConstraint::new(Interval::<Second>::from_f64(0.0, 100.0));
        let leaf = ConstraintNode::leaf(constraint);

        assert!(leaf.is_leaf());
        assert_eq!(leaf.depth(), 1);
        assert_eq!(leaf.leaf_count(), 1);
        assert_eq!(leaf.node_count(), 1);
    }

    #[test]
    fn test_intersection_node() {
        let child1 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 100.0,
        )));
        let child2 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            50.0, 150.0,
        )));

        let intersection = ConstraintNode::intersection(vec![child1, child2]);

        assert!(intersection.is_intersection());
        assert_eq!(intersection.depth(), 2);
        assert_eq!(intersection.leaf_count(), 2);
        assert_eq!(intersection.node_count(), 3);
    }

    #[test]
    fn test_union_node() {
        let child1 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 50.0,
        )));
        let child2 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            100.0, 150.0,
        )));

        let union = ConstraintNode::union(vec![child1, child2]);

        assert!(union.is_union());
        assert_eq!(union.depth(), 2);
        assert_eq!(union.leaf_count(), 2);
        assert_eq!(union.node_count(), 3);
    }

    #[test]
    fn test_flatten() {
        let a = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 100.0,
        )));
        let b = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            50.0, 150.0,
        )));

        let nested = ConstraintNode::intersection(vec![ConstraintNode::intersection(vec![a, b])]);

        let flattened = nested.flatten();

        assert_eq!(flattened.leaf_count(), 2);
        // After flattening, should be a single intersection with 2 children
        assert!(flattened.is_intersection());
        assert_eq!(flattened.children().unwrap().len(), 2);
    }
}
