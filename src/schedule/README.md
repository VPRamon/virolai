# Schedule Module

An efficient, production-ready schedule data structure for managing non-overlapping time intervals (tasks).

## Overview

The `Schedule` struct maintains a collection of non-overlapping intervals indexed by task ID, providing:
- Fast insertion with automatic conflict detection
- Efficient lookup by task ID or time position
- Quick conflict queries for proposed intervals
- Iteration in chronological order

## Features

### Core Operations
- **Add tasks**: O(log n) insertion with automatic overlap checking
- **Remove tasks**: O(log n) removal by task ID
- **Get interval**: O(1) hash + O(log n) tree lookup
- **Find conflicts**: O(log n + k) where k is the number of conflicts
- **Task at time**: O(log n) lookup of which task is active at a given time

### Additional Capabilities
- Iterate over tasks in chronological order
- Calculate statistics (total duration, span, earliest/latest times)
- Check if time slots are free
- Comprehensive error handling with informative error messages

## Architecture

### Module Structure
```
schedule/
├── mod.rs         # Main Schedule implementation
├── entry_key.rs   # F64Key and Entry types for internal storage
├── errors.rs      # Error types with Display and Error traits
└── tests.rs       # Comprehensive test suite (42 tests)
```

### Internal Design

The schedule uses a dual-index approach:
1. **BTreeMap by start time**: Enables chronological iteration and efficient range queries
2. **HashMap by task ID**: Enables O(1) lookup of task metadata

This design ensures:
- Only adjacent intervals need to be checked for overlaps during insertion
- Conflicts can be found by scanning from the first potentially overlapping task
- Task removal is efficient even though we store by start time

### Key Types

#### `F64Key`
Total-order wrapper for `f64` using IEEE-754 total ordering, enabling use as `BTreeMap` keys.
Rejects NaN values to maintain schedule integrity.

#### `Entry<U>`
Internal storage type mapping task IDs to intervals with accessor methods.

#### `ScheduleError`
Comprehensive error enumeration:
- `DuplicateTaskId`: Task ID already exists
- `NaNTime`: Invalid NaN time value
- `OverlapsExisting`: New interval conflicts with existing task
- `TaskNotFound`: Task ID not found (future extension)

## Usage Examples

### Basic Usage

```rust
use v_rolai::schedule::Schedule;
use v_rolai::solution_space::Interval;
use qtty::{Quantity, Second};

let mut schedule = Schedule::<Second>::new();

// Add non-overlapping tasks
schedule.add(1, Interval::from_f64(0.0, 10.0))?;
schedule.add(2, Interval::from_f64(15.0, 25.0))?;

// Check for conflicts
let query = Interval::from_f64(8.0, 20.0);
let conflicts = schedule.conflicts_vec(query)?;
println!("Found {} conflicts", conflicts.len());

// Find task at specific time
if let Some(id) = schedule.task_at(Quantity::new(5.0))? {
    println!("Task {} is active at t=5.0", id);
}
```

### Conflict Detection

```rust
// Check if a slot is free
let proposal = Interval::from_f64(10.5, 14.5);
if schedule.is_free(proposal)? {
    schedule.add(3, proposal)?;
    println!("Task added successfully");
}

// Get detailed conflict information
let conflicts = schedule.conflicts_vec(proposal)?;
for (id, interval) in conflicts {
    println!("Conflicts with task {}: {}", id, interval);
}
```

### Statistics and Analysis

```rust
println!("Total tasks: {}", schedule.len());
println!("Total duration: {:.1}s", schedule.total_duration().value());
println!("Earliest start: {:.1}s", schedule.earliest_start().unwrap().value());
println!("Latest end: {:.1}s", schedule.latest_end().unwrap().value());
println!("Total span: {:.1}s", schedule.span().unwrap().value());
```

### Iteration

```rust
// Iterate in chronological order
for (id, interval) in schedule.iter() {
    println!("Task {}: {}", id, interval);
}

// Get all task IDs
let ids: Vec<_> = schedule.task_ids().collect();

// Get all intervals
let intervals: Vec<_> = schedule.intervals().collect();
```

## Testing

The module includes a comprehensive test suite with 42 tests covering:
- ✅ Basic operations (add, remove, get, clear)
- ✅ Overlap detection (touching, contained, containing intervals)
- ✅ Conflict queries (single, multiple, none)
- ✅ Task lookup by time position
- ✅ Iterator functionality and ordering
- ✅ Statistics calculations
- ✅ Edge cases (zero duration, many tasks, large values, negative times)
- ✅ NaN and infinity handling

Run tests:
```bash
cargo test --lib schedule::
```

Run with output:
```bash
cargo test --lib schedule:: -- --nocapture
```

## Examples

See the complete example:
```bash
cargo run --example schedule_usage
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `new()` | O(1) | Creates empty schedule |
| `add()` | O(log n) | Checks only neighbors |
| `remove()` | O(log n) | Hash lookup + tree removal |
| `get_interval()` | O(log n) | Hash + tree lookup |
| `contains_task()` | O(1) | Hash lookup only |
| `conflicts()` | O(log n + k) | k = number of conflicts |
| `task_at()` | O(log n) | Binary search by time |
| `iter()` | O(1) start | Iterator creation |

Space complexity: O(n) where n is the number of tasks

## Design Decisions

### Why BTreeMap + HashMap?
- **BTreeMap**: Maintains chronological order, enables range queries
- **HashMap**: Provides O(1) task ID lookups
- Trade-off: Slight memory overhead for significant performance gains

### Why Inclusive Intervals?
The schedule uses inclusive endpoints `[start, end]` consistent with the `Interval` type.
This means intervals `[0, 10]` and `[10, 20]` are considered overlapping at point 10.

### Why Reject NaN?
NaN violates total ordering requirements and represents invalid time values.
Early rejection prevents undefined behavior and maintains schedule integrity.

## Future Enhancements

Potential extensions (not yet implemented):
- Serialization support (serde)
- Task priorities or metadata
- Bulk operations (add multiple tasks)
- Gap analysis utilities
- Time-based queries (tasks in range)
- Optimization hints (e.g., pre-reserve capacity)

## Related Modules

- `solution_space::Interval`: The interval type used by the schedule
- `scheduling_block`: Higher-level scheduling constructs
- `constraints`: Temporal and logical constraints for scheduling

## License

Same as parent project.
