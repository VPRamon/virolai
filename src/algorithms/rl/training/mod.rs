//! Training infrastructure for MAPPO (Multi-Agent PPO).
//!
//! Provides rollout buffer, GAE computation, and the MAPPO trainer.

pub mod buffer;
pub mod gae;
pub mod mappo;

pub use mappo::{MAPPOTrainer, TrainingConfig};
