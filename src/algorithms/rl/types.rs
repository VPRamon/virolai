//! Core types for the RL scheduling algorithm.
//!
//! Defines agent types, spatial positions, and coalition requirements
//! used throughout the multi-agent reinforcement learning system.

use std::fmt;

/// Agent age group, which determines maximum movement speed.
///
/// Speed hierarchy: `Young > Middle > Old`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentType {
    Young,
    Middle,
    Old,
}

impl AgentType {
    /// Returns the default maximum speed for this agent type.
    ///
    /// These are reference values; actual speeds are configured via [`super::config::RLConfig`].
    pub fn default_max_speed(&self) -> f64 {
        match self {
            AgentType::Young => 3.0,
            AgentType::Middle => 2.0,
            AgentType::Old => 1.0,
        }
    }

    /// Returns all agent types in order.
    pub fn all() -> [AgentType; 3] {
        [AgentType::Young, AgentType::Middle, AgentType::Old]
    }

    /// Returns the index of this type (0=Young, 1=Middle, 2=Old).
    pub fn index(&self) -> usize {
        match self {
            AgentType::Young => 0,
            AgentType::Middle => 1,
            AgentType::Old => 2,
        }
    }

    /// One-hot encoding of this agent type as a 3-element vector.
    pub fn one_hot(&self) -> [f64; 3] {
        let mut v = [0.0; 3];
        v[self.index()] = 1.0;
        v
    }
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentType::Young => write!(f, "young"),
            AgentType::Middle => write!(f, "middle"),
            AgentType::Old => write!(f, "old"),
        }
    }
}

/// Minimum agent-type requirements for collecting a task.
///
/// A task is collected when at least `young` young agents, `middle` middle-aged agents,
/// and `old` old agents are simultaneously within the collection radius.
///
/// Excess agents are allowed and never penalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentTypeRequirements {
    pub young: u32,
    pub middle: u32,
    pub old: u32,
}

impl AgentTypeRequirements {
    /// Creates a new requirements specification.
    pub fn new(young: u32, middle: u32, old: u32) -> Self {
        Self { young, middle, old }
    }

    /// Returns true if the given agent counts satisfy all requirements.
    ///
    /// Excess agents (counts exceeding requirements) are always acceptable.
    pub fn is_satisfied_by(&self, young_count: u32, middle_count: u32, old_count: u32) -> bool {
        young_count >= self.young && middle_count >= self.middle && old_count >= self.old
    }

    /// Returns the requirement for a specific agent type.
    pub fn requirement_for(&self, agent_type: AgentType) -> u32 {
        match agent_type {
            AgentType::Young => self.young,
            AgentType::Middle => self.middle,
            AgentType::Old => self.old,
        }
    }

    /// Total number of agents required across all types.
    pub fn total(&self) -> u32 {
        self.young + self.middle + self.old
    }

    /// Returns requirements as an array `[young, middle, old]`.
    pub fn as_array(&self) -> [u32; 3] {
        [self.young, self.middle, self.old]
    }
}

impl Default for AgentTypeRequirements {
    fn default() -> Self {
        Self {
            young: 1,
            middle: 0,
            old: 0,
        }
    }
}

/// A 2D position in the environment domain Ω = [0, W] × [0, H].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    /// Creates a new position.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Origin position (0, 0).
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Euclidean distance to another position.
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the unit direction vector from `self` toward `target`.
    ///
    /// Returns `(0, 0)` if positions are coincident.
    pub fn direction_to(&self, target: &Position) -> (f64, f64) {
        let dx = target.x - self.x;
        let dy = target.y - self.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-12 {
            (0.0, 0.0)
        } else {
            (dx / dist, dy / dist)
        }
    }

    /// Clamps this position to lie within the domain [0, width] × [0, height].
    pub fn clamp_to_bounds(&mut self, width: f64, height: f64) {
        self.x = self.x.clamp(0.0, width);
        self.y = self.y.clamp(0.0, height);
    }

    /// Returns a new position clamped to the domain bounds.
    pub fn clamped(mut self, width: f64, height: f64) -> Self {
        self.clamp_to_bounds(width, height);
        self
    }

    /// Normalizes position to [0, 1] × [0, 1] given domain dimensions.
    pub fn normalized(&self, width: f64, height: f64) -> (f64, f64) {
        (self.x / width, self.y / height)
    }

    /// Moves toward `target` by at most `max_dist`, clamped to domain bounds.
    pub fn move_toward(&mut self, target: &Position, max_dist: f64, width: f64, height: f64) {
        let (dx, dy) = self.direction_to(target);
        let dist = self.distance_to(target);
        let step = dist.min(max_dist);
        self.x += dx * step;
        self.y += dy * step;
        self.clamp_to_bounds(width, height);
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_type_speeds_are_ordered() {
        assert!(AgentType::Young.default_max_speed() > AgentType::Middle.default_max_speed());
        assert!(AgentType::Middle.default_max_speed() > AgentType::Old.default_max_speed());
    }

    #[test]
    fn agent_type_one_hot() {
        assert_eq!(AgentType::Young.one_hot(), [1.0, 0.0, 0.0]);
        assert_eq!(AgentType::Middle.one_hot(), [0.0, 1.0, 0.0]);
        assert_eq!(AgentType::Old.one_hot(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn requirements_satisfied_exact() {
        let req = AgentTypeRequirements::new(2, 1, 0);
        assert!(req.is_satisfied_by(2, 1, 0));
    }

    #[test]
    fn requirements_satisfied_excess() {
        let req = AgentTypeRequirements::new(2, 1, 0);
        assert!(req.is_satisfied_by(5, 3, 2)); // excess is fine
    }

    #[test]
    fn requirements_not_satisfied() {
        let req = AgentTypeRequirements::new(2, 1, 0);
        assert!(!req.is_satisfied_by(1, 1, 0)); // too few young
    }

    #[test]
    fn position_distance() {
        let a = Position::new(0.0, 0.0);
        let b = Position::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn position_direction_to() {
        let a = Position::new(0.0, 0.0);
        let b = Position::new(1.0, 0.0);
        let (dx, dy) = a.direction_to(&b);
        assert!((dx - 1.0).abs() < 1e-10);
        assert!(dy.abs() < 1e-10);
    }

    #[test]
    fn position_clamp() {
        let p = Position::new(-1.0, 15.0).clamped(10.0, 10.0);
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 10.0);
    }

    #[test]
    fn position_move_toward() {
        let mut p = Position::new(0.0, 0.0);
        let target = Position::new(10.0, 0.0);
        p.move_toward(&target, 3.0, 20.0, 20.0);
        assert!((p.x - 3.0).abs() < 1e-10);
        assert!(p.y.abs() < 1e-10);
    }

    #[test]
    fn position_move_toward_clamps() {
        let mut p = Position::new(9.0, 0.0);
        let target = Position::new(20.0, 0.0);
        p.move_toward(&target, 5.0, 10.0, 10.0);
        assert_eq!(p.x, 10.0); // clamped
    }
}
