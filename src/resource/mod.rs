//! Resource abstraction for instrument-level constraints.
//!
//! Resources represent shared scheduling constraints that apply across multiple tasks,
//! such as observatory-level constraints (nighttime, moon altitude) that are computed
//! once and then intersected with each task's solution space.
//!
//! # Motivation
//!
//! Instead of duplicating instrument constraints (like "astronomical night") on every task,
//! we compute them once at the resource level and intersect task windows with the
//! resource's availability windows.

mod instrument;
mod resource;

pub use instrument::Instrument;
pub use resource::Resource;
