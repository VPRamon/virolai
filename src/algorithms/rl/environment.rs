//! RL scheduling environment.
//!
//! Implements the simulation loop described in §8 of the algorithm design:
//! spawn → act → move → tick → collect → expire → reward → transition.

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::agent::AgentState;
use super::config::RLConfig;
use super::observation::ObservationBuilder;
use super::reward::RewardComputer;
use super::task_pool::{TaskPool, TaskTemplate};
use super::types::{AgentType, Position};

/// Result of a single environment step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Per-agent observations after the step.
    pub observations: Vec<Vec<f64>>,
    /// Cooperative reward (shared across all agents).
    pub reward: f64,
    /// Whether the episode is done (horizon reached).
    pub done: bool,
    /// Current time step.
    pub time_step: u32,
    /// Number of tasks collected this step.
    pub tasks_collected: usize,
    /// Number of tasks expired this step.
    pub tasks_expired: usize,
    /// Global state (for centralized critic, if needed).
    pub global_state: Vec<f64>,
}

/// The multi-agent RL scheduling environment.
///
/// Simulates a 2D domain with N heterogeneous agents collecting time-limited
/// tasks that have minimum agent-type requirements. Follows the cooperative
/// Markov game formulation from §3.
///
/// # Lifecycle
///
/// 1. Call [`RLEnvironment::new`] with configuration and seed.
/// 2. Call [`RLEnvironment::reset`] to initialize an episode.
/// 3. Repeatedly call [`RLEnvironment::step`] with agent actions until `done`.
/// 4. Inspect [`StepResult`] for rewards, observations, and episode status.
#[derive(Debug)]
pub struct RLEnvironment {
    /// Environment configuration.
    pub config: RLConfig,
    /// Agent states.
    pub agents: Vec<AgentState>,
    /// Task pool.
    pub task_pool: TaskPool,
    /// Current time step.
    pub t: u32,
    /// Random number generator.
    rng: StdRng,
    /// Seed for reproducible resets.
    seed: u64,
    /// Cumulative reward this episode.
    pub cumulative_reward: f64,
}

impl RLEnvironment {
    /// Creates a new environment with the given configuration and RNG seed.
    ///
    /// # Arguments
    ///
    /// * `config` - Environment and reward configuration
    /// * `seed` - Random seed for reproducible episodes
    pub fn new(config: RLConfig, seed: u64) -> Self {
        Self {
            config,
            agents: Vec::new(),
            task_pool: TaskPool::with_defaults(),
            t: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            cumulative_reward: 0.0,
        }
    }

    /// Creates a new environment with custom task templates.
    pub fn with_templates(config: RLConfig, templates: Vec<TaskTemplate>, seed: u64) -> Self {
        Self {
            config,
            agents: Vec::new(),
            task_pool: TaskPool::new(templates),
            t: 0,
            rng: StdRng::seed_from_u64(seed),
            seed,
            cumulative_reward: 0.0,
        }
    }

    /// Sets the agent composition for this environment.
    ///
    /// # Arguments
    ///
    /// * `composition` - Slice of `(count, AgentType)` pairs.
    ///   e.g., `&[(2, AgentType::Young), (2, AgentType::Middle), (2, AgentType::Old)]`
    pub fn set_agents(&mut self, composition: &[(u32, AgentType)]) {
        self.agents.clear();
        let mut idx = 0u32;
        for (count, agent_type) in composition {
            for _ in 0..*count {
                let position = Position::new(
                    self.config.world_width / 2.0,
                    self.config.world_height / 2.0,
                );
                self.agents.push(AgentState::new(
                    format!("agent_{}", idx),
                    position,
                    *agent_type,
                ));
                idx += 1;
            }
        }
    }

    /// Resets the environment for a new episode.
    ///
    /// Repositions all agents to the center, clears the task pool,
    /// and returns initial observations.
    pub fn reset(&mut self) -> Vec<Vec<f64>> {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.seed += 1; // different seed each episode
        self.t = 0;
        self.cumulative_reward = 0.0;

        // Reset agent positions to center
        let cx = self.config.world_width / 2.0;
        let cy = self.config.world_height / 2.0;
        for agent in &mut self.agents {
            agent.position = Position::new(cx, cy);
            agent.current_target = 0;
            agent.prev_distance_to_target = None;
        }

        // Reset task pool
        self.task_pool.reset();

        // Initial spawn
        self.task_pool.spawn(&mut self.rng, &self.config);

        // Build initial observations
        ObservationBuilder::build_all(&self.agents, &self.task_pool, &self.config)
    }

    /// Executes one environment step.
    ///
    /// Follows the loop from §8:
    /// 1. Spawn tasks
    /// 2. Agents choose actions (provided as input)
    /// 3. Move agents toward targets
    /// 4. Update task timers
    /// 5. Collect tasks (check coalitions)
    /// 6. Expire tasks
    /// 7. Compute reward (with shaping)
    /// 8. Build next observations
    ///
    /// # Arguments
    ///
    /// * `actions` - One action per agent. Action 0 = patrol, 1..=M = target task index.
    pub fn step(&mut self, actions: Vec<usize>) -> StepResult {
        assert_eq!(
            actions.len(),
            self.agents.len(),
            "Number of actions must match number of agents"
        );

        // 1. Spawn new tasks
        self.task_pool.spawn(&mut self.rng, &self.config);

        // 2-3. Agents choose actions and move
        let agent_positions: Vec<_> = self.agents.iter().map(|a| a.position).collect();
        let top_m = self.task_pool.top_m(&agent_positions, self.config.top_m);
        let top_m_refs: Vec<&_> = top_m.to_vec();

        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.step(actions[i], &top_m_refs, &self.config);
        }

        // 4. Update task timers
        self.task_pool.tick();

        // 5. Collect tasks
        let agent_info: Vec<_> = self
            .agents
            .iter()
            .map(|a| (a.position, a.agent_type))
            .collect();
        let (collected, collected_value) = self.task_pool.try_collect(&agent_info);

        // 6. Expire tasks
        let expired = self.task_pool.expire();

        // 7. Compute reward
        // Recompute top_m after collection/expiration for progress measurement
        let agent_positions_after: Vec<_> = self.agents.iter().map(|a| a.position).collect();
        let top_m_after = self
            .task_pool
            .top_m(&agent_positions_after, self.config.top_m);
        let top_m_after_refs: Vec<&_> = top_m_after.to_vec();

        let reward = RewardComputer::compute(
            collected_value,
            &expired,
            &self.agents,
            &top_m_after_refs,
            &self.config,
        );
        self.cumulative_reward += reward;

        // Advance time
        self.t += 1;
        let done = self.t >= self.config.episode_horizon;

        // 8. Build next observations
        let observations =
            ObservationBuilder::build_all(&self.agents, &self.task_pool, &self.config);
        let global_state =
            ObservationBuilder::build_global_state(&self.agents, &self.task_pool, &self.config);

        StepResult {
            observations,
            reward,
            done,
            time_step: self.t,
            tasks_collected: collected.len(),
            tasks_expired: expired.len(),
            global_state,
        }
    }

    /// Returns the number of agents.
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Returns the number of currently active tasks.
    pub fn n_active_tasks(&self) -> usize {
        self.task_pool.active.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env() -> RLEnvironment {
        let config = RLConfig::default();
        let mut env = RLEnvironment::new(config, 42);
        env.set_agents(&[
            (2, AgentType::Young),
            (2, AgentType::Middle),
            (2, AgentType::Old),
        ]);
        env
    }

    #[test]
    fn reset_returns_observations() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 6); // 6 agents
        for o in &obs {
            assert_eq!(o.len(), env.config.observation_dim());
        }
    }

    #[test]
    fn step_returns_valid_result() {
        let mut env = make_env();
        env.reset();
        let actions = vec![0; env.n_agents()]; // all patrol
        let result = env.step(actions);
        assert_eq!(result.observations.len(), env.n_agents());
        assert_eq!(result.time_step, 1);
        assert!(!result.done);
    }

    #[test]
    fn episode_terminates_at_horizon() {
        let mut env = make_env();
        env.config.episode_horizon = 5;
        env.reset();
        for t in 0..5 {
            let actions = vec![0; env.n_agents()];
            let result = env.step(actions);
            if t < 4 {
                assert!(!result.done);
            } else {
                assert!(result.done);
            }
        }
    }

    #[test]
    fn agents_initialized_at_center() {
        let mut env = make_env();
        env.reset();
        let cx = env.config.world_width / 2.0;
        let cy = env.config.world_height / 2.0;
        for agent in &env.agents {
            assert!((agent.position.x - cx).abs() < 1e-10);
            assert!((agent.position.y - cy).abs() < 1e-10);
        }
    }
}
