//! RL-based scheduling module.
//!
//! The full multi-agent RL environment, policies, and training infrastructure
//! are behind the `rl` feature flag (which brings in `rand`).  Neural network
//! policies and MAPPO training additionally require the `rl-nn` feature flag.
//!
//! The [`RLScheduler`] (feature `rl-nn`) bridges a trained policy to the
//! [`SchedulingAlgorithm`](crate::algorithms::SchedulingAlgorithm) trait.

// Always available — no extra dependencies.
pub mod types;

// Modules that require the `rl` feature (rand dependency).
#[cfg(feature = "rl")]
pub mod agent;
#[cfg(feature = "rl")]
pub mod config;
#[cfg(feature = "rl")]
pub mod environment;
#[cfg(feature = "rl")]
pub mod metrics;
#[cfg(feature = "rl")]
pub mod observation;
#[cfg(feature = "rl")]
pub mod policy;
#[cfg(feature = "rl")]
pub mod reward;
#[cfg(feature = "rl")]
pub mod task_pool;

#[cfg(feature = "rl-nn")]
pub mod network;
#[cfg(feature = "rl-nn")]
pub mod policy_scheduler;
#[cfg(feature = "rl-nn")]
pub mod training;

// Public re-exports — always available.
pub use types::{AgentType, AgentTypeRequirements, Position};

// Re-exports gated behind `rl` feature.
#[cfg(feature = "rl")]
pub use agent::AgentState;
#[cfg(feature = "rl")]
pub use config::RLConfig;
#[cfg(feature = "rl")]
pub use environment::{RLEnvironment, StepResult};
#[cfg(feature = "rl")]
pub use metrics::EvaluationMetrics;
#[cfg(feature = "rl")]
pub use observation::ObservationBuilder;
#[cfg(feature = "rl")]
pub use policy::{GreedyHeuristicPolicy, Policy, RandomPolicy};
#[cfg(feature = "rl")]
pub use reward::RewardComputer;
#[cfg(feature = "rl")]
pub use task_pool::{TaskInstance, TaskPool};

#[cfg(feature = "rl-nn")]
pub use network::NeuralPolicy;
#[cfg(feature = "rl-nn")]
pub use policy_scheduler::RLScheduler;
#[cfg(feature = "rl-nn")]
pub use training::{MAPPOTrainer, TrainingConfig};
