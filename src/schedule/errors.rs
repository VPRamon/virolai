use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    /// Task ID is already present in the schedule
    DuplicateTaskId(u64),
    /// A time value was NaN, which is not allowed
    NaNTime,
    /// New interval overlaps with an existing interval
    OverlapsExisting { new_id: u64, existing_id: u64 },
    /// Task ID was not found in the schedule
    TaskNotFound(u64),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::DuplicateTaskId(id) => {
                write!(f, "Task ID {} already exists in schedule", id)
            }
            ScheduleError::NaNTime => {
                write!(f, "Time value cannot be NaN")
            }
            ScheduleError::OverlapsExisting {
                new_id,
                existing_id,
            } => {
                write!(
                    f,
                    "Task {} overlaps with existing task {}",
                    new_id, existing_id
                )
            }
            ScheduleError::TaskNotFound(id) => {
                write!(f, "Task ID {} not found in schedule", id)
            }
        }
    }
}

impl std::error::Error for ScheduleError {}
