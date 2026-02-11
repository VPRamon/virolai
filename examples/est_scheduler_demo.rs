//! Demonstration of the Early Starting Time (EST) scheduling algorithm.

use qtty::{Quantity, Second};
use vrolai::algorithms::{ESTScheduler, SchedulingAlgorithm};
use vrolai::constraints::IntervalConstraint;
use vrolai::scheduling_block::{SchedulingBlock, Task};
use vrolai::solution_space::{Interval, SolutionSpace};

#[derive(Debug, Clone)]
struct SimpleTask {
    name: String,
    duration: Quantity<Second>,
    priority: i32,
}

impl SimpleTask {
    fn new(name: &str, duration_sec: f64, priority: i32) -> Self {
        Self {
            name: name.to_string(),
            duration: Quantity::new(duration_sec),
            priority,
        }
    }
}

impl Task<Second> for SimpleTask {
    type SizeUnit = Second;
    type ConstraintLeaf = IntervalConstraint<Second>;

    fn name(&self) -> &str {
        &self.name
    }

    fn size(&self) -> Quantity<Second> {
        self.duration
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

fn main() {
    // Create scheduling blocks with tasks
    let mut block = SchedulingBlock::<SimpleTask, Second>::new();

    // Add tasks with different priorities and durations
    let task1 = SimpleTask::new("High Priority Task", 300.0, 10);
    let task2 = SimpleTask::new("Medium Priority Task", 200.0, 5);
    let task3 = SimpleTask::new("Low Priority Task", 150.0, 1);
    let task4 = SimpleTask::new("Another Medium Task", 250.0, 5);

    block.add_task(task1);
    block.add_task(task2);
    block.add_task(task3);
    block.add_task(task4);

    // Define the scheduling horizon (e.g., 24 hours = 86400 seconds)
    let horizon = Interval::new(
        Quantity::<Second>::new(0.0),
        Quantity::<Second>::new(1000.0),
    );

    // Create solution space (all tasks can be scheduled anywhere in the horizon)
    let solution_space = SolutionSpace::populate(&[block.clone()], horizon);

    // Create EST scheduler with endangered threshold of 100 seconds
    let scheduler = ESTScheduler::new(100);

    // Run the scheduler
    let schedule = scheduler.schedule(&[block], &solution_space, horizon);

    // Display results
    println!("EST Scheduling Results:");
    println!("======================");
    println!("Total tasks scheduled: {}", schedule.len());
    println!();

    for (task_id, interval) in schedule.iter() {
        println!(
            "Task {}: [{:.1}s - {:.1}s] (duration: {:.1}s)",
            task_id,
            interval.start().value(),
            interval.end().value(),
            interval.duration().value()
        );
    }
}
