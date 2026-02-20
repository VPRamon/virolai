//! Task instance management: spawning, collection, and expiration.

use rand::Rng;

use super::config::RLConfig;
use super::types::{AgentType, AgentTypeRequirements, Position};
use crate::Id;

/// A concrete task instance active in the RL environment.
///
/// Unlike the abstract [`crate::scheduling_block::Task`] trait, this is a
/// runtime data structure representing a specific task that has appeared at
/// a location and is counting down to its deadline.
#[derive(Debug, Clone)]
pub struct TaskInstance {
    /// Unique identifier.
    pub id: Id,
    /// Position in the 2D domain.
    pub position: Position,
    /// Reward value for collection.
    pub value: f64,
    /// Original deadline (total steps before expiration).
    pub deadline: u32,
    /// Remaining steps before expiration.
    pub remaining_time: u32,
    /// Minimum agent-type requirements for collection.
    pub type_requirements: AgentTypeRequirements,
    /// Collection radius ρ for this task.
    pub collection_radius: f64,
    /// Remaining appearances for this task type (None = unlimited).
    pub remaining_appearances: Option<u32>,
}

impl TaskInstance {
    /// Returns this task's features as a vector for observation encoding.
    ///
    /// Format: `[x_norm, y_norm, value_norm, time_left_norm, r_young, r_middle, r_old, k_rem_norm, heading_young, heading_middle]`
    ///
    /// Values are normalized by the provided scales.
    pub fn features(
        &self,
        config: &RLConfig,
        max_value: f64,
        heading_counts: [u32; 3],
    ) -> Vec<f64> {
        let (nx, ny) = self
            .position
            .normalized(config.world_width, config.world_height);
        let v_norm = if max_value > 0.0 {
            self.value / max_value
        } else {
            0.0
        };
        let t_norm = self.remaining_time as f64 / config.episode_horizon as f64;
        let reqs = self.type_requirements.as_array();
        let k_norm = self
            .remaining_appearances
            .map(|k| k as f64 / 10.0)
            .unwrap_or(1.0);

        vec![
            nx,
            ny,
            v_norm,
            t_norm,
            reqs[0] as f64,
            reqs[1] as f64,
            reqs[2] as f64,
            k_norm,
            heading_counts[0] as f64,
            heading_counts[1] as f64,
        ]
    }
}

/// A task specification template used for spawning new instances.
#[derive(Debug, Clone)]
pub struct TaskTemplate {
    /// Base name (instances get numbered suffixes).
    pub name: String,
    /// Reward value range [min, max].
    pub value_range: (f64, f64),
    /// Deadline range [min, max] in time steps.
    pub deadline_range: (u32, u32),
    /// Agent-type requirements.
    pub type_requirements: AgentTypeRequirements,
    /// Collection radius override (None = use global default).
    pub collection_radius: Option<f64>,
    /// Maximum total appearances (None = unlimited).
    pub max_appearances: Option<u32>,
    /// Appearances used so far.
    pub appearances_used: u32,
}

impl TaskTemplate {
    /// Returns true if this template has remaining appearances.
    pub fn can_spawn(&self) -> bool {
        match self.max_appearances {
            Some(max) => self.appearances_used < max,
            None => true,
        }
    }
}

/// Manages the pool of active, collected, and expired tasks.
#[derive(Debug, Clone)]
pub struct TaskPool {
    /// Currently active tasks.
    pub active: Vec<TaskInstance>,
    /// IDs of tasks collected this episode.
    pub collected_ids: Vec<Id>,
    /// IDs of tasks that expired this episode.
    pub expired_ids: Vec<Id>,
    /// Task templates for spawning.
    pub templates: Vec<TaskTemplate>,
    /// Running counter for unique task IDs.
    next_id: u32,
}

impl TaskPool {
    /// Creates a new empty task pool with the given templates.
    pub fn new(templates: Vec<TaskTemplate>) -> Self {
        Self {
            active: Vec::new(),
            collected_ids: Vec::new(),
            expired_ids: Vec::new(),
            templates,
            next_id: 0,
        }
    }

    /// Creates a task pool with default templates suitable for demos.
    pub fn with_defaults() -> Self {
        let templates = vec![
            TaskTemplate {
                name: "easy".into(),
                value_range: (1.0, 3.0),
                deadline_range: (15, 30),
                type_requirements: AgentTypeRequirements::new(1, 0, 0),
                collection_radius: None,
                max_appearances: None,
                appearances_used: 0,
            },
            TaskTemplate {
                name: "medium".into(),
                value_range: (3.0, 7.0),
                deadline_range: (10, 20),
                type_requirements: AgentTypeRequirements::new(1, 1, 0),
                collection_radius: None,
                max_appearances: Some(10),
                appearances_used: 0,
            },
            TaskTemplate {
                name: "hard".into(),
                value_range: (7.0, 15.0),
                deadline_range: (8, 15),
                type_requirements: AgentTypeRequirements::new(2, 1, 1),
                collection_radius: None,
                max_appearances: Some(5),
                appearances_used: 0,
            },
        ];
        Self::new(templates)
    }

    /// Resets the pool for a new episode.
    pub fn reset(&mut self) {
        self.active.clear();
        self.collected_ids.clear();
        self.expired_ids.clear();
        self.next_id = 0;
        for t in &mut self.templates {
            t.appearances_used = 0;
        }
    }

    /// Spawns new tasks stochastically based on spawn rate λ.
    ///
    /// At each call, for each template with remaining appearances, a task
    /// is spawned with probability `config.spawn_rate`, up to the active limit.
    pub fn spawn<R: Rng>(&mut self, rng: &mut R, config: &RLConfig) {
        if self.active.len() >= config.max_active_tasks {
            return;
        }

        // Collect spawnable template indices
        let spawnable: Vec<usize> = self
            .templates
            .iter()
            .enumerate()
            .filter(|(_, t)| t.can_spawn())
            .map(|(i, _)| i)
            .collect();

        for idx in spawnable {
            if self.active.len() >= config.max_active_tasks {
                break;
            }
            if rng.gen::<f64>() > config.spawn_rate {
                continue;
            }

            let template = &mut self.templates[idx];
            let value = rng.gen::<f64>() * (template.value_range.1 - template.value_range.0)
                + template.value_range.0;
            let deadline = rng.gen_range(template.deadline_range.0..=template.deadline_range.1);
            let position = Position::new(
                rng.gen::<f64>() * config.world_width,
                rng.gen::<f64>() * config.world_height,
            );
            let radius = template
                .collection_radius
                .unwrap_or(config.collection_radius);

            let id = format!("{}_{}", template.name, self.next_id);
            self.next_id += 1;
            template.appearances_used += 1;

            self.active.push(TaskInstance {
                id,
                position,
                value,
                deadline,
                remaining_time: deadline,
                type_requirements: template.type_requirements,
                collection_radius: radius,
                remaining_appearances: template
                    .max_appearances
                    .map(|max| max - template.appearances_used),
            });
        }
    }

    /// Decrements remaining time for all active tasks.
    pub fn tick(&mut self) {
        for task in &mut self.active {
            task.remaining_time = task.remaining_time.saturating_sub(1);
        }
    }

    /// Checks which tasks can be collected and removes them from active.
    ///
    /// A task is collected when agents of each required type within the
    /// collection radius meet or exceed the minimum counts.
    ///
    /// # Returns
    ///
    /// `(collected_tasks, total_value)` — the collected tasks and sum of their values.
    pub fn try_collect(
        &mut self,
        agent_positions: &[(Position, AgentType)],
    ) -> (Vec<TaskInstance>, f64) {
        let mut collected = Vec::new();
        let mut total_value = 0.0;

        self.active.retain(|task| {
            // Count agents by type within collection radius
            let mut counts = [0u32; 3];
            for (pos, atype) in agent_positions {
                if pos.distance_to(&task.position) <= task.collection_radius {
                    counts[atype.index()] += 1;
                }
            }

            if task
                .type_requirements
                .is_satisfied_by(counts[0], counts[1], counts[2])
            {
                total_value += task.value;
                collected.push(task.clone());
                false // remove from active
            } else {
                true // keep
            }
        });

        for t in &collected {
            self.collected_ids.push(t.id.clone());
        }

        (collected, total_value)
    }

    /// Removes and returns all expired tasks (remaining_time == 0).
    pub fn expire(&mut self) -> Vec<TaskInstance> {
        let mut expired = Vec::new();

        self.active.retain(|task| {
            if task.remaining_time == 0 {
                expired.push(task.clone());
                false
            } else {
                true
            }
        });

        for t in &expired {
            self.expired_ids.push(t.id.clone());
        }

        expired
    }

    /// Returns the Top-M candidate tasks scored by urgency-weighted value/distance.
    ///
    /// Scoring: `score(j) = V_j / (ε + min_i dist(p_i, q_j)) × 1/(ε + d_j)`
    ///
    /// # Arguments
    ///
    /// * `agent_positions` - Positions of all agents (for distance computation)
    /// * `m` - Number of candidates to return
    pub fn top_m(&self, agent_positions: &[Position], m: usize) -> Vec<&TaskInstance> {
        let eps = 1e-6;
        let mut scored: Vec<(f64, &TaskInstance)> = self
            .active
            .iter()
            .map(|task| {
                let min_dist = agent_positions
                    .iter()
                    .map(|p| p.distance_to(&task.position))
                    .fold(f64::INFINITY, f64::min);
                let score =
                    (task.value / (eps + min_dist)) * (1.0 / (eps + task.remaining_time as f64));
                (score, task)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(m).map(|(_, t)| t).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> TaskPool {
        TaskPool::with_defaults()
    }

    #[test]
    fn spawn_respects_max_active() {
        let mut pool = make_pool();
        let config = RLConfig {
            max_active_tasks: 2,
            spawn_rate: 1.0, // always spawn
            ..RLConfig::default()
        };
        let mut rng = rand::thread_rng();
        // Spawn multiple times
        for _ in 0..10 {
            pool.spawn(&mut rng, &config);
        }
        assert!(pool.active.len() <= 2);
    }

    #[test]
    fn tick_decrements_time() {
        let mut pool = TaskPool::new(vec![]);
        pool.active.push(TaskInstance {
            id: "t0".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 5,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        });
        pool.tick();
        assert_eq!(pool.active[0].remaining_time, 4);
    }

    #[test]
    fn collect_when_requirements_met() {
        let mut pool = TaskPool::new(vec![]);
        pool.active.push(TaskInstance {
            id: "t0".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 5,
            type_requirements: AgentTypeRequirements::new(1, 0, 0),
            collection_radius: 2.0,
            remaining_appearances: None,
        });

        // Agent within radius
        let agents = vec![(Position::new(5.0, 5.0), AgentType::Young)];
        let (collected, value) = pool.try_collect(&agents);
        assert_eq!(collected.len(), 1);
        assert!((value - 10.0).abs() < 1e-10);
        assert!(pool.active.is_empty());
    }

    #[test]
    fn no_collect_when_requirements_not_met() {
        let mut pool = TaskPool::new(vec![]);
        pool.active.push(TaskInstance {
            id: "t0".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 5,
            type_requirements: AgentTypeRequirements::new(2, 0, 0), // need 2 young
            collection_radius: 2.0,
            remaining_appearances: None,
        });

        // Only 1 young agent
        let agents = vec![(Position::new(5.0, 5.0), AgentType::Young)];
        let (collected, _) = pool.try_collect(&agents);
        assert!(collected.is_empty());
        assert_eq!(pool.active.len(), 1);
    }

    #[test]
    fn expire_removes_timed_out() {
        let mut pool = TaskPool::new(vec![]);
        pool.active.push(TaskInstance {
            id: "t0".into(),
            position: Position::new(5.0, 5.0),
            value: 10.0,
            deadline: 1,
            remaining_time: 0,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        });

        let expired = pool.expire();
        assert_eq!(expired.len(), 1);
        assert!(pool.active.is_empty());
        assert_eq!(pool.expired_ids.len(), 1);
    }

    #[test]
    fn top_m_scores_correctly() {
        let mut pool = TaskPool::new(vec![]);
        // Close high-value task
        pool.active.push(TaskInstance {
            id: "close".into(),
            position: Position::new(1.0, 0.0),
            value: 10.0,
            deadline: 5,
            remaining_time: 2,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        });
        // Far low-value task
        pool.active.push(TaskInstance {
            id: "far".into(),
            position: Position::new(100.0, 100.0),
            value: 1.0,
            deadline: 5,
            remaining_time: 20,
            type_requirements: AgentTypeRequirements::default(),
            collection_radius: 1.0,
            remaining_appearances: None,
        });

        let agents = vec![Position::new(0.0, 0.0)];
        let top = pool.top_m(&agents, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, "close"); // higher score
    }
}
