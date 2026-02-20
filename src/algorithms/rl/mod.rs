//! Multi-agent Reinforcement Learning scheduling algorithm.
//!
//! This module implements a cooperative multi-agent RL system for scheduling
//! tasks in a 2D spatial environment with heterogeneous agents and coalition
//! requirements. The approach uses hierarchical control: RL decides task
//! assignment (high-level) while a deterministic controller handles movement
//! (low-level).
//!
//! # Architecture
//!
//! - **Environment** ([`environment::RLEnvironment`]): Simulates agents moving in a
//!   2D domain, collecting tasks that spawn and expire over time.
//! - **Policies**: Pluggable via the [`policy::Policy`] trait. Includes a
//!   [`policy::RandomPolicy`] for testing and a [`policy::GreedyHeuristicPolicy`]
//!   as a competitive baseline.
//! - **Observation**: Per-agent observations encode own state plus Top-M candidate
//!   tasks, built by [`observation::ObservationBuilder`].
//! - **Reward**: Composite reward function combining collection value, shaping,
//!   and penalties, computed by [`reward::RewardComputer`].
//!
//! # Feature flags
//!
//! - `rl-nn`: Enables neural network policies and MAPPO training via `tch-rs`.
//!   Without this flag, only the environment, heuristic policies, and evaluation
//!   tools are available.
//!
//! # Quick start
//!
//! ```ignore
//! use virolai::algorithms::rl::{RLConfig, RLEnvironment, GreedyHeuristicPolicy, EvaluationMetrics};
//!
//! let config = RLConfig::default();
//! let mut env = RLEnvironment::new(config, 42);
//! let mut policy = GreedyHeuristicPolicy::new();
//! let metrics = EvaluationMetrics::evaluate(&mut env, &mut policy, 10);
//! println!("{}", metrics);
//! ```

pub mod agent;
pub mod config;
pub mod environment;
pub mod metrics;
pub mod observation;
pub mod policy;
pub mod reward;
pub mod task_pool;
pub mod types;

#[cfg(feature = "rl-nn")]
pub mod network;
#[cfg(feature = "rl-nn")]
pub mod training;

// Public re-exports
pub use agent::AgentState;
pub use config::RLConfig;
pub use environment::{RLEnvironment, StepResult};
pub use metrics::EvaluationMetrics;
pub use observation::ObservationBuilder;
pub use policy::{GreedyHeuristicPolicy, Policy, RandomPolicy};
pub use reward::RewardComputer;
pub use task_pool::{TaskInstance, TaskPool};
pub use types::{AgentType, AgentTypeRequirements, Position};

#[cfg(feature = "rl-nn")]
pub use network::NeuralPolicy;
#[cfg(feature = "rl-nn")]
pub use training::{MAPPOTrainer, TrainingConfig};
