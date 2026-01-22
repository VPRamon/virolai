//! Example demonstrating the Schedule module's capabilities.
//!
//! Run with: `cargo run --example schedule_usage`

use qtty::{Quantity, Second};
use vrolai::schedule::Schedule;
use vrolai::solution_space::Interval;

type Time = Quantity<Second>;

fn main() {
    println!("=== Schedule Module Usage Example ===\n");

    // Create a new schedule
    let mut schedule = Schedule::<Second>::new();
    println!("Created empty schedule");

    // Add some tasks
    println!("\n--- Adding Tasks ---");
    add_task(&mut schedule, "1", 0.0, 10.0, "Data Download");
    add_task(&mut schedule, "2", 15.0, 25.0, "Image Processing");
    add_task(&mut schedule, "3", 30.0, 45.0, "Transmission");
    add_task(&mut schedule, "4", 50.0, 55.0, "Calibration");

    println!("\nTotal tasks: {}", schedule.len());
    println!(
        "Total scheduled time: {:.1} seconds",
        schedule.total_duration().value()
    );
    println!(
        "Schedule span: {:.1} seconds (from {} to {})",
        schedule.span().unwrap().value(),
        schedule.earliest_start().unwrap().value(),
        schedule.latest_end().unwrap().value()
    );

    // Display all scheduled tasks
    println!("\n--- Scheduled Tasks (in order) ---");
    for (id, interval) in schedule.iter() {
        println!(
            "Task {}: {} (duration: {:.1}s)",
            id,
            interval,
            interval.duration().value()
        );
    }

    // Try to add an overlapping task
    println!("\n--- Attempting to Add Overlapping Task ---");
    let result = schedule.add("5", Interval::from_f64(12.0, 20.0));
    match result {
        Ok(_) => println!("Task 5 added successfully"),
        Err(e) => println!("Failed to add task 5: {}", e),
    }

    // Check for conflicts
    println!("\n--- Conflict Detection ---");
    let query1 = Interval::from_f64(8.0, 12.0);
    println!("Checking interval {}", query1);
    match schedule.conflicts_vec(query1) {
        Ok(conflicts) => {
            if conflicts.is_empty() {
                println!("  No conflicts - slot is free!");
            } else {
                println!("  Conflicts with {} task(s):", conflicts.len());
                for (id, interval) in conflicts {
                    println!("    - Task {} {}", id, interval);
                }
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    let query2 = Interval::from_f64(10.5, 14.5);
    println!("Checking interval {}", query2);
    if schedule.is_free(query2).unwrap() {
        println!("  Slot is free!");
    } else {
        println!("  Slot has conflicts");
    }

    // Find task at specific times
    println!("\n--- Task Lookup by Time ---");
    let test_times = vec![5.0, 12.0, 20.0, 32.0, 60.0];
    for time in test_times {
        match schedule.task_at(Time::new(time)).unwrap() {
            Some(id) => {
                let interval = schedule.get_interval(&id).unwrap();
                println!("Time {:.1}s -> Task {} {}", time, id, interval);
            }
            None => println!("Time {:.1}s -> No task scheduled", time),
        }
    }

    // Remove a task and show the gap
    println!("\n--- Removing Task ---");
    if let Some(removed) = schedule.remove("2") {
        println!("Removed task 2: {}", removed);
        println!("Remaining tasks: {}", schedule.len());
    }

    println!("\n--- Updated Schedule ---");
    for (id, interval) in schedule.iter() {
        println!("Task {}: {}", id, interval);
    }

    // Demonstrate adding to the freed slot
    println!("\n--- Adding Task to Freed Slot ---");
    let new_task = Interval::from_f64(16.0, 24.0);
    match schedule.add("5", new_task) {
        Ok(_) => println!("Successfully added task 5: {}", new_task),
        Err(e) => println!("Failed: {}", e),
    }

    // Complex scenario: find all free slots
    println!("\n--- Finding Free Time Slots ---");
    find_free_slots(&schedule, 0.0, 60.0, 5.0);

    // Statistics
    println!("\n--- Final Statistics ---");
    println!("Total tasks: {}", schedule.len());
    println!(
        "Total scheduled duration: {:.1}s",
        schedule.total_duration().value()
    );
    println!(
        "Schedule efficiency: {:.1}%",
        (schedule.total_duration().value() / schedule.span().unwrap_or(Time::new(1.0)).value())
            * 100.0
    );

    println!("\n=== Example Complete ===");
}

fn add_task(schedule: &mut Schedule<Second>, id: &str, start: f64, end: f64, name: &str) {
    let interval = Interval::from_f64(start, end);
    match schedule.add(id, interval) {
        Ok(_) => println!("✓ Added task {} '{}': {}", id, name, interval),
        Err(e) => println!("✗ Failed to add task {}: {}", id, e),
    }
}

fn find_free_slots(schedule: &Schedule<Second>, start: f64, end: f64, min_duration: f64) {
    println!(
        "Searching for free slots of at least {:.1}s between {:.1}s and {:.1}s:",
        min_duration, start, end
    );

    let mut current = start;
    let mut free_slots = Vec::new();

    // Iterate through scheduled tasks to find gaps
    for (_id, interval) in schedule.iter() {
        let task_start = interval.start().value();
        let task_end = interval.end().value();

        if task_start > current {
            let gap_duration = task_start - current;
            if gap_duration >= min_duration {
                free_slots.push((current, task_start, gap_duration));
            }
        }
        current = current.max(task_end);
    }

    // Check for free time after last task
    if end > current {
        let gap_duration = end - current;
        if gap_duration >= min_duration {
            free_slots.push((current, end, gap_duration));
        }
    }

    if free_slots.is_empty() {
        println!("  No free slots found");
    } else {
        for (slot_start, slot_end, duration) in free_slots {
            println!(
                "  [{:.1}, {:.1}] - duration: {:.1}s",
                slot_start, slot_end, duration
            );
        }
    }
}
