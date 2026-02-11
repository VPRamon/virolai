//! Example demonstrating SolutionSpace with task references
//!
//! This example shows how SolutionSpace now holds references to tasks instead of string IDs,
//! allowing type-safe task identification and cross-block scheduling.

use qtty::{Quantity, Second};
use vrolai::constraints::{ConstraintExpr, IntervalConstraint};
use vrolai::scheduling_block::{SchedulingBlock, Task};
use vrolai::solution_space::{Interval, SolutionSpace};

#[derive(Debug)]
struct MyTask {
    name: String,
    size: Quantity<Second>,
    constraints: Option<ConstraintExpr<IntervalConstraint<Second>>>,
}

impl Task<Second> for MyTask {
    type SizeUnit = Second;
    type ConstraintLeaf = IntervalConstraint<Second>;

    fn name(&self) -> &str {
        &self.name
    }

    fn size(&self) -> Quantity<Second> {
        self.size
    }

    fn constraints(&self) -> Option<&ConstraintExpr<IntervalConstraint<Second>>> {
        self.constraints.as_ref()
    }
}

fn main() {
    println!("🔍 Task-based Solution Space Example\n");

    // Create two scheduling blocks
    let mut block1: SchedulingBlock<MyTask> = SchedulingBlock::new();
    let mut block2: SchedulingBlock<MyTask> = SchedulingBlock::new();

    // Task with constraints
    let task_a = MyTask {
        name: "TaskA".to_string(),
        size: Quantity::<Second>::new(100.0),
        constraints: Some(ConstraintExpr::leaf(IntervalConstraint::new(
            Interval::from_f64(0.0, 500.0),
        ))),
    };

    // Task without constraints - will get the full scheduling range
    let task_b = MyTask {
        name: "TaskB".to_string(),
        size: Quantity::<Second>::new(50.0),
        constraints: None,
    };

    // Task from another block with different constraints
    let task_c = MyTask {
        name: "TaskC".to_string(),
        size: Quantity::<Second>::new(75.0),
        constraints: Some(ConstraintExpr::leaf(IntervalConstraint::new(
            Interval::from_f64(600.0, 1000.0),
        ))),
    };

    let task_a_id = block1.add_task(task_a);
    let task_b_id = block1.add_task(task_b);
    let task_c_id = block2.add_task(task_c);

    // Populate solution space from multiple blocks
    let range = Interval::new(
        Quantity::<Second>::new(0.0),
        Quantity::<Second>::new(1000.0),
    );

    let blocks = vec![block1, block2];
    let solution_space = SolutionSpace::populate(&blocks, range);

    println!("📊 Solution Space Summary:");
    println!("  Tasks: {}", solution_space.count());
    println!("  Total intervals: {}", solution_space.interval_count());
    println!(
        "  Total capacity: {:.1}s\n",
        solution_space.total_capacity().value()
    );

    println!("🔎 Per-Task Queries:\n");

    // Query TaskA (has constraints)
    if let Some(intervals) = solution_space.get_intervals(&task_a_id) {
        println!("TaskA (id={}):", task_a_id);
        println!("  Intervals: {}", intervals.len());
        println!(
            "  Capacity: {:.1}s",
            solution_space.capacity(&task_a_id).value()
        );
        println!(
            "  Can place at t=50s? {}",
            solution_space.can_place(&task_a_id, Quantity::new(50.0), Quantity::new(100.0))
        );
        println!(
            "  Can place at t=600s? {}",
            solution_space.can_place(&task_a_id, Quantity::new(600.0), Quantity::new(100.0))
        );
    }

    // Query TaskB (no constraints - has full range)
    if let Some(intervals) = solution_space.get_intervals(&task_b_id) {
        println!("\nTaskB (id={}, no constraints):", task_b_id);
        println!("  Intervals: {}", intervals.len());
        println!(
            "  Capacity: {:.1}s",
            solution_space.capacity(&task_b_id).value()
        );
        for (i, interval) in intervals.iter().enumerate() {
            println!(
                "  [{}] [{:.1}s, {:.1}s]",
                i,
                interval.start().value(),
                interval.end().value()
            );
        }
    }

    // Query TaskC (from different block)
    if let Some(intervals) = solution_space.get_intervals(&task_c_id) {
        println!("\nTaskC (id={}):", task_c_id);
        println!("  Intervals: {}", intervals.len());
        println!(
            "  Earliest fit: {:.1}s",
            solution_space
                .find_earliest_fit_for(&task_c_id, Quantity::new(75.0))
                .map(|q| q.value())
                .unwrap_or(f64::NAN)
        );
    }

    println!("\n📋 Task IDs in solution space:");
    for task_id in solution_space.ids() {
        println!("  - id: {}", task_id);
    }

    println!("\n✨ Full Solution Space:");
    println!("{}", solution_space);
}
