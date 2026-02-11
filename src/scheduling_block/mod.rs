pub mod error;
pub mod task;

mod block;
pub use block::SchedulingBlock;

pub use error::SchedulingError;
pub use task::Task;

// Re-export from the dedicated `resource` module for backward compatibility.
#[deprecated(note = "Use `vrolai::resource::Resource` instead")]
pub use crate::resource::Resource;
