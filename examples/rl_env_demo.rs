// Demonstration: run the RL environment and evaluate a baseline policy.
//
// Build/run from this repo root:
//   cargo run --manifest-path dev-deps/virolai/Cargo.toml --features rl --example rl_env_demo -- --policy greedy --episodes 50

#[cfg(feature = "rl")]
fn main() {
    use std::env;

    use virolai::algorithms::rl::{
        AgentType, EvaluationMetrics, GreedyHeuristicPolicy, Policy, RLConfig, RLEnvironment,
        RandomPolicy,
    };

    let args: Vec<String> = env::args().collect();
    let policy_name = arg_value(&args, "--policy").unwrap_or("greedy");
    let episodes: usize = arg_value(&args, "--episodes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let config = RLConfig::default();
    let mut env = RLEnvironment::new(config.clone(), seed);
    env.set_agents(&[(2, AgentType::Young), (2, AgentType::Middle), (1, AgentType::Old)]);

    let mut policy: Box<dyn Policy> = match policy_name {
        "random" => Box::new(RandomPolicy::new(config.action_dim())),
        "greedy" => Box::new(GreedyHeuristicPolicy::new(config.clone())),
        other => {
            eprintln!("Unknown --policy '{}'; expected 'greedy' or 'random'.", other);
            std::process::exit(2);
        }
    };

    let metrics = EvaluationMetrics::evaluate(&mut env, policy.as_mut(), episodes);
    println!("Policy: {}", policy.name());
    println!("{}", metrics);
}

#[cfg(not(feature = "rl"))]
fn main() {
    eprintln!(
        "This example requires the 'rl' feature.\n\
Run:\n\
  cargo run --manifest-path dev-deps/virolai/Cargo.toml --features rl --example rl_env_demo -- --policy greedy --episodes 50"
    );
}

#[cfg(feature = "rl")]
fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

