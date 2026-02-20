//! RL scheduler demo: multi-agent cooperative task collection.
//!
//! Demonstrates the RL scheduling environment with heterogeneous agents
//! (young, middle, old) collecting time-limited tasks in a 2D domain.
//!
//! Run:
//! ```sh
//! cargo run --example rl_scheduler_demo
//! ```
//!
//! With neural network training (requires libtorch):
//! ```sh
//! cargo run --example rl_scheduler_demo --features rl-nn
//! ```

use virolai::algorithms::rl::{
    AgentType, EvaluationMetrics, GreedyHeuristicPolicy, Policy, RLConfig, RLEnvironment,
    RandomPolicy,
};

fn main() {
    println!("=== Multi-Agent RL Scheduler Demo ===\n");

    // Configure environment
    let config = RLConfig {
        world_width: 10.0,
        world_height: 10.0,
        episode_horizon: 50,
        spawn_rate: 0.4,
        collection_radius: 1.5,
        top_m: 5,
        ..RLConfig::default()
    };

    // Create environment with 6 agents: 2 young, 2 middle, 2 old
    let mut env = RLEnvironment::new(config.clone(), 42);
    env.set_agents(&[
        (2, AgentType::Young),
        (2, AgentType::Middle),
        (2, AgentType::Old),
    ]);

    println!("Environment:");
    println!(
        "  Domain: {}×{}",
        env.config.world_width, env.config.world_height
    );
    println!("  Agents: {} (2 young, 2 middle, 2 old)", env.n_agents());
    println!("  Horizon: {} steps", env.config.episode_horizon);
    println!("  Top-M candidates: {}", env.config.top_m);
    println!("  Collection radius: {}", env.config.collection_radius);
    println!();

    // --- Evaluate Random Policy ---
    println!("Evaluating Random Policy...");
    let mut random_policy = RandomPolicy::new(config.action_dim());
    let random_metrics = EvaluationMetrics::evaluate(&mut env, &mut random_policy, 20);
    println!("{}", random_metrics);

    // --- Evaluate Greedy Heuristic Policy ---
    println!("Evaluating Greedy Heuristic Policy...");
    let mut heuristic_policy = GreedyHeuristicPolicy::new(config.clone());
    let heuristic_metrics = EvaluationMetrics::evaluate(&mut env, &mut heuristic_policy, 20);
    println!("{}", heuristic_metrics);

    // --- Run a single episode with heuristic policy for detailed output ---
    println!("--- Detailed Single Episode (Heuristic) ---\n");
    let obs = env.reset();
    let mut step = 0;
    let mut total_collected = 0;
    let mut total_expired = 0;
    let mut obs = obs;

    loop {
        let actions = heuristic_policy.select_actions(&obs);
        let result = env.step(actions);

        if result.tasks_collected > 0 || result.tasks_expired > 0 {
            println!(
                "  Step {:3}: collected={}, expired={}, active_tasks={}, reward={:.3}",
                result.time_step,
                result.tasks_collected,
                result.tasks_expired,
                env.n_active_tasks(),
                result.reward,
            );
        }

        total_collected += result.tasks_collected;
        total_expired += result.tasks_expired;
        obs = result.observations;
        step += 1;

        if result.done {
            break;
        }
    }

    println!();
    println!("Episode summary:");
    println!("  Steps: {}", step);
    println!("  Tasks collected: {}", total_collected);
    println!("  Tasks expired: {}", total_expired);
    println!("  Cumulative reward: {:.3}", env.cumulative_reward);

    // --- Neural network training (behind feature flag) ---
    #[cfg(feature = "rl-nn")]
    {
        use tch::Device;
        use virolai::algorithms::rl::{MAPPOTrainer, NeuralPolicy, TrainingConfig};

        println!("\n=== MAPPO Training ===\n");

        let train_config = TrainingConfig {
            n_episodes_per_update: 4,
            n_epochs: 2,
            ..TrainingConfig::default()
        };

        let mut trainer =
            MAPPOTrainer::new(config.clone(), train_config, env.n_agents(), Device::Cpu);

        let curve = trainer.train(&mut env, 50);

        println!("\nLearning curve:");
        for (update, reward) in &curve {
            if update % 10 == 0 {
                println!("  Update {:3}: mean_reward = {:.3}", update, reward);
            }
        }
    }
}
