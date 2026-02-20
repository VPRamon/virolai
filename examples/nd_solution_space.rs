//! Demonstrates a 2-dimensional solution space (time × distance).
//!
//! This example shows how to build a `SolutionSpaceND<(Second, Meter)>` where
//! each task has independent feasibility windows on a *time* axis and a
//! *distance* (range) axis.  The EST engine is not used here — the purpose is
//! to show the data-layer API for multi-axis feasibility queries.
//!
//! Run with:
//!
//! ```text
//! cargo run --example nd_solution_space
//! ```

use std::fmt::Debug;

use qtty::{Meter, Quantity, Second};
use virolai::scheduling_block::TaskND;
use virolai::solution_space::{Interval, IntervalSet, Region2, SolutionSpaceND};

// ─────────────────────────────────────────────────────────────────────────────
// 2-D task definition
// ─────────────────────────────────────────────────────────────────────────────

/// A task that is feasible within a window in both time and distance.
#[derive(Debug)]
struct SpacetimeTask {
    name: String,
    /// Duration on the primary (time) axis.
    duration: Quantity<Second>,
}

impl TaskND<(Second, Meter)> for SpacetimeTask {
    fn name(&self) -> &str {
        &self.name
    }

    fn size_on_primary(&self) -> Quantity<Second> {
        self.duration
    }
}

fn main() {
    // ── Build a 2-D solution space ────────────────────────────────────

    let mut space: SolutionSpaceND<(Second, Meter)> = SolutionSpaceND::new();

    // Task A: available 0–100 s, reachable at 200–500 m
    let time_a = IntervalSet::from(vec![Interval::<Second>::from_f64(0.0, 100.0)]);
    let dist_a = IntervalSet::from(vec![Interval::<Meter>::from_f64(200.0, 500.0)]);
    space.add_region("task-a", Region2::new(time_a, dist_a));

    // Task B: two time windows, one distance band
    let time_b = IntervalSet::from(vec![
        Interval::<Second>::from_f64(10.0, 40.0),
        Interval::<Second>::from_f64(60.0, 90.0),
    ]);
    let dist_b = IntervalSet::from(vec![Interval::<Meter>::from_f64(0.0, 300.0)]);
    space.add_region("task-b", Region2::new(time_b, dist_b));

    // Task C: only a primary-axis window (secondary axis unconstrained → empty)
    let time_c = IntervalSet::from(vec![Interval::<Second>::from_f64(50.0, 150.0)]);
    space.add_region(
        "task-c",
        Region2::<Second, Meter>::new(
            time_c,
            IntervalSet::new(), // no distance constraint
        ),
    );

    println!("Space has {} entries", space.count());

    // Demonstrate TaskND — log the task names via the trait
    let tasks: Vec<Box<dyn TaskND<(Second, Meter)>>> = vec![
        Box::new(SpacetimeTask {
            name: "task-a".to_string(),
            duration: Quantity::new(30.0),
        }),
        Box::new(SpacetimeTask {
            name: "task-b".to_string(),
            duration: Quantity::new(20.0),
        }),
    ];
    for t in &tasks {
        println!(
            "  '{}': primary size = {:.1} s",
            t.name(),
            t.size_on_primary().value()
        );
    }

    // ── Generic (N-D) queries ─────────────────────────────────────────

    // Access full region
    let region_b = space.get_region("task-b").unwrap();
    println!(
        "task-b primary windows: {} interval(s)",
        region_b.axis_0.len()
    );
    println!(
        "task-b distance windows: {} interval(s)",
        region_b.axis_1.len()
    );

    // Access primary axis only (works for any arity via get_primary_intervals)
    let primary_a = space.get_primary_intervals("task-a").unwrap();
    println!(
        "task-a primary axis intervals: {:?}",
        primary_a
            .iter()
            .map(|i| (i.start().value(), i.end().value()))
            .collect::<Vec<_>>()
    );

    // ── Mutable access to a region ────────────────────────────────────

    if let Some(region) = space.get_region_mut("task-c") {
        // Add a distance constraint to task-c after the fact
        region
            .axis_1
            .push(Interval::<Meter>::from_f64(100.0, 400.0));
        println!("task-c now has {} distance window(s)", region.axis_1.len());
    }

    // ── 1-D backward-compatible alias ────────────────────────────────
    //
    // Existing 1-D code using `SolutionSpace<U>` is unchanged.  Shown here
    // for illustration: populate a classic 1-D space alongside the 2-D one.

    use virolai::solution_space::SolutionSpace;

    let mut classic: SolutionSpace<Second> = SolutionSpace::new();
    classic.add_interval("task-x", Interval::from_f64(0.0, 50.0));
    println!(
        "classic 1-D space has {} entries, {} intervals total",
        classic.count(),
        classic.interval_count()
    );

    // 1-D space is SolutionSpaceND<(Second,)>, so get_region() also works:
    let region_x = classic.get_region("task-x").unwrap();
    println!(
        "task-x via get_region: {} interval(s)",
        region_x.len() // IntervalSet<Second>: Deref<Target=[Interval<Second>]>
    );

    println!("Example complete.");
}
