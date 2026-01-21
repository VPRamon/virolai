//! Composable constraint trees with AND/OR logic.
use crate::constraints::constraint::Constraint;
use crate::solution_space::Interval;
use qtty::Unit;
use std::ops::Not;
use std::sync::Arc;

/// Constraint tree node: leaf (concrete constraint) or combinator (AND/OR).
///
/// Trees enable composing complex scheduling logic:
/// - **Leaf**: Wraps a concrete [`Constraint`]
/// - **Intersection**: AND logic – all children must be satisfied
/// - **Union**: OR logic – at least one child must be satisfied
///
/// Note: Uses `Arc` for atomic reference-counted ownership to enable
/// thread-safe tree sharing and avoid deep cloning.
#[derive(Debug, Clone)]
pub enum ConstraintNode<U: Unit> {
    Leaf(Arc<dyn Constraint<U>>),
    Not(Arc<ConstraintNode<U>>),
    Intersection(Vec<ConstraintNode<U>>),
    Union(Vec<ConstraintNode<U>>),
}

impl<U: Unit> ConstraintNode<U> {
    pub fn leaf(constraint: impl Constraint<U> + 'static) -> Self {
        ConstraintNode::Leaf(Arc::new(constraint))
    }

    /// Creates intersection node (AND logic).
    pub fn intersection(children: Vec<ConstraintNode<U>>) -> Self {
        ConstraintNode::Intersection(children)
    }

    /// Creates union node (OR logic).
    pub fn union(children: Vec<ConstraintNode<U>>) -> Self {
        ConstraintNode::Union(children)
    }

    pub fn stringify(&self) -> String {
        match self {
            ConstraintNode::Leaf(constraint) => constraint.stringify(),
            ConstraintNode::Not(child) => format!("Not({})", child.stringify()),
            ConstraintNode::Intersection(children) => format!(
                "Intersection({})",
                children
                    .iter()
                    .map(|c| c.stringify())
                    .collect::<Vec<_>>()
                    .join(" ∩ ")
            ),
            ConstraintNode::Union(children) => format!(
                "Union({})",
                children
                    .iter()
                    .map(|c| c.stringify())
                    .collect::<Vec<_>>()
                    .join(" ∪ ")
            ),
        }
    }

    /// Recursively evaluates tree, returning valid intervals.
    ///
    /// - Leaf: delegates to wrapped constraint
    /// - Intersection: computes overlap of all child intervals
    /// - Union: merges all child intervals
    pub fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>> {
        match self {
            ConstraintNode::Leaf(constraint) => constraint.compute_intervals(range),
            ConstraintNode::Not(child) => {
                super::operations::compute_complement(child.compute_intervals(range), range)
            }
            ConstraintNode::Intersection(children) => children
                .iter()
                .map(|c| c.compute_intervals(range))
                .reduce(|acc, v| super::operations::compute_intersection(&acc, &v))
                .unwrap_or_default(),
            ConstraintNode::Union(children) => children
                .iter()
                .map(|c| c.compute_intervals(range))
                .fold(Vec::new(), |acc, v| {
                    super::operations::compute_union(&acc, &v)
                }),
        }
    }

    /// Prints the tree structure in a human-readable format.
    ///
    /// # Arguments
    ///
    /// * `indent` - The current indentation level (use 0 for the root)
    pub fn print_tree(&self, indent: usize) {
        let prefix = "  ".repeat(indent);
        match self {
            ConstraintNode::Leaf(constraint) => {
                println!("{}└─ Leaf: {}", prefix, constraint.stringify());
            }
            ConstraintNode::Not(child) => {
                println!("{}└─ Not", prefix);
                child.print_tree(indent + 1);
            }
            ConstraintNode::Intersection(children) => {
                println!("{}└─ Intersection", prefix);
                for child in children {
                    child.print_tree(indent + 1);
                }
            }
            ConstraintNode::Union(children) => {
                println!("{}└─ Union", prefix);
                for child in children {
                    child.print_tree(indent + 1);
                }
            }
        }
    }
}

impl<U: Unit> Constraint<U> for ConstraintNode<U> {
    fn compute_intervals(&self, range: Interval<U>) -> Vec<Interval<U>> {
        ConstraintNode::compute_intervals(self, range)
    }

    fn stringify(&self) -> String {
        ConstraintNode::stringify(self)
    }
}

impl<U: Unit> Not for ConstraintNode<U> {
    type Output = Self;

    fn not(self) -> Self {
        ConstraintNode::Not(Arc::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::IntervalConstraint;
    use qtty::{Quantity, Second};
    use Interval;

    #[test]
    fn test_intersection_node() {
        let child1 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 100.0,
        )));
        let child2 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            50.0, 150.0,
        )));

        let intersection = ConstraintNode::intersection(vec![child1, child2]);

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
        let child1 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            0.0, 50.0,
        )));
        let child2 = ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
            100.0, 150.0,
        )));

        let union = ConstraintNode::union(vec![child1, child2]);

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
        let a = ConstraintNode::leaf(IntervalConstraint::new(Interval::from_f64(0.0, 100.0)));
        let b = ConstraintNode::leaf(IntervalConstraint::new(Interval::from_f64(50.0, 150.0)));
        let c = ConstraintNode::leaf(IntervalConstraint::new(Interval::from_f64(200.0, 300.0)));
        let d = ConstraintNode::leaf(IntervalConstraint::new(Interval::from_f64(250.0, 350.0)));

        let intersection1 = ConstraintNode::intersection(vec![a, b]);
        let intersection2 = ConstraintNode::intersection(vec![c, d]);
        let union = ConstraintNode::union(vec![intersection1, intersection2]);

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
        let forbidden = ConstraintNode::union(vec![
            ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                20.0, 40.0,
            ))),
            ConstraintNode::leaf(IntervalConstraint::new(Interval::<Second>::from_f64(
                60.0, 80.0,
            ))),
        ]);

        let not = ConstraintNode::not(forbidden);

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
    fn test_constraint_trait_impl_does_not_recurse() {
        let tree = ConstraintNode::leaf(IntervalConstraint::new(Interval::from_f64(0.0, 10.0)));

        let as_trait: Arc<dyn Constraint<Second>> = Arc::new(tree);
        let intervals = as_trait.compute_intervals(Interval::from_f64(0.0, 100.0));
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].start().value(), 0.0);
        assert_eq!(intervals[0].end().value(), 10.0);
    }
}
