# VIROLAI
Versatile Instrumentation Resource Optimization Leveraging AI

A Rust library for DAG-based task scheduling using `petgraph`, designed for flexibility, type safety, and extensibility.

## Features

- **Trait-based Task Abstraction**: Define custom task types by implementing the `Task` trait
- **Generic DAG Structure**: Type-safe scheduling blocks with any task and dependency types
- **Cycle Prevention**: Proactive cycle detection ensures DAG invariants
- **Rich Query APIs**: Topological ordering, critical path analysis, root/leaf detection
- **Thread-Safe**: All tasks are `Send + Sync + 'static` for concurrent scheduling
- **Object-Safe**: Can use trait objects (`Box<dyn Task>`) when needed

## Quick Start

### 1. Define Your Task Types

```rust
use virolai::scheduling_block::Task;

#[derive(Debug, Clone)]
struct MyTask {
    name: String,
    duration_ms: u64,
}

impl Task for MyTask {
    fn name(&self) -> &str {
        &self.name
    }

    fn duration_ms(&self) -> u64 {
        self.duration_ms
    }
}
```

### 2. Create a Scheduling Block

```rust
use virolai::scheduling_block::{SchedulingBlock, SchedulingError};

fn main() -> Result<(), SchedulingError> {
    let mut schedule = SchedulingBlock::new();
    
    // Add tasks
    let task1 = schedule.add_task(MyTask { name: "Task 1".into(), duration_ms: 1000 });
    let task2 = schedule.add_task(MyTask { name: "Task 2".into(), duration_ms: 2000 });
    let task3 = schedule.add_task(MyTask { name: "Task 3".into(), duration_ms: 1500 });
    
    // Define dependencies (task1 must complete before task2 and task3)
    schedule.add_dependency(task1, task2, ())?;
    schedule.add_dependency(task1, task3, ())?;
    
    // Get execution order
    let order = schedule.topo_order()?;
    for node in order {
        let task = schedule.get_task(node).unwrap();
        println!("Execute: {}", task.name());
    }
    
    // Compute critical path
    let (duration, path) = schedule.critical_path()?;
    println!("Total duration: {}ms", duration);
    
    Ok(())
}
```

### 3. Mix Multiple Task Types

Use enums to combine different task types:

```rust
enum TaskType {
    Observation(ObservationTask),
    Calibration(CalibrationTask),
}

impl Task for TaskType {
    fn name(&self) -> &str {
        match self {
            TaskType::Observation(t) => t.name(),
            TaskType::Calibration(t) => t.name(),
        }
    }
    
    fn duration_ms(&self) -> u64 {
        match self {
            TaskType::Observation(t) => t.duration_ms(),
            TaskType::Calibration(t) => t.duration_ms(),
        }
    }
}

let mut schedule = SchedulingBlock::<TaskType, DependencyKind>::new();
```

## Examples

Astronomy-specific examples now live in `astro_scheduler`.

```bash
# Astronomical observation task scheduling
cargo run -p astro_scheduler --example astro_observation

# Altitude constraint demos
cargo run -p astro_scheduler --example altitude_constraint_demo
cargo run -p astro_scheduler --example altitude_multi_target

# Demo application using VIROLAI + astronomy modules
cargo run -p astro_scheduler
```

See `astro_scheduler/examples/README.md`.

## Documentation

- **[DESIGN.md](DESIGN.md)**: Detailed design rationale and trade-offs
- **[API Docs](https://docs.rs/virolai)**: Generated API documentation

## API Overview

### Core Trait

```rust
pub trait Task: Send + Sync + Debug + 'static {
    fn name(&self) -> &str;
    fn duration_ms(&self) -> u64;
    fn priority(&self) -> Option<i32> { None }
    fn can_parallelize(&self) -> bool { true }
}
```

### SchedulingBlock Methods

- `new()` - Create empty scheduling block
- `add_task(task: T)` - Add a task, returns NodeIndex
- `add_dependency(from, to, dep)` - Add dependency with cycle detection
- `topo_order()` - Get topological ordering
- `critical_path()` - Compute longest path and total duration
- `roots()` - Get tasks with no dependencies
- `leaves()` - Get tasks with no dependents
- `graph()` - Access underlying petgraph for advanced queries

### Error Handling

```rust
pub enum SchedulingError {
    CycleDetected,
    InvalidNodeIndex(NodeIndex),
    GraphContainsCycle,
    EmptyGraph,
}
```

## Design Philosophy

VIROLAI uses **generic types** (`SchedulingBlock<T: Task, D>`) rather than trait objects (`Box<dyn Task>`) as the primary pattern because:

- ✅ Zero-cost abstraction (no heap allocation or virtual dispatch)
- ✅ Type safety (access task-specific fields without downcasting)
- ✅ Better performance (compiler optimizations, cache locality)
- ✅ Natural enum-based composition in Rust

Trait objects are still supported for dynamic use cases (plugins, runtime loading).

See [DESIGN.md](DESIGN.md) for detailed rationale.

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please open an issue or PR.
