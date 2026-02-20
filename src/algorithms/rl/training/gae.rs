//! Generalized Advantage Estimation (GAE-λ).
//!
//! Computes advantages and returns from rollout data using the
//! centralized value function.

/// Computes GAE-λ advantages and discounted returns.
///
/// # Arguments
///
/// * `rewards` - Per-step rewards
/// * `values` - Per-step value estimates from the critic
/// * `dones` - Per-step episode termination flags
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE λ parameter (0 = TD(0), 1 = Monte Carlo)
///
/// # Returns
///
/// `(advantages, returns)` where `returns = advantages + values`.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    assert_eq!(values.len(), n);
    assert_eq!(dones.len(), n);

    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;

    for t in (0..n).rev() {
        let next_value = if t + 1 < n { values[t + 1] } else { 0.0 };
        let next_non_terminal = if dones[t] { 0.0 } else { 1.0 };

        let delta = rewards[t] + gamma * next_value * next_non_terminal - values[t];
        gae = delta + gamma * gae_lambda * next_non_terminal * gae;
        advantages[t] = gae;
    }

    let returns: Vec<f64> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_basic() {
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, true];
        let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.99, 0.95);

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);

        // Last step (done=true): delta = 1.0 + 0 - 0.5 = 0.5, gae = 0.5
        assert!((advantages[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn gae_with_lambda_zero() {
        // λ=0 => TD(0): advantages are just TD errors
        let rewards = vec![1.0, 2.0];
        let values = vec![0.5, 1.0];
        let dones = vec![false, true];
        let gamma = 0.99;
        let (advantages, _) = compute_gae(&rewards, &values, &dones, gamma, 0.0);

        // t=1 (done=true): delta = 2.0 + 0 - 1.0 = 1.0
        assert!((advantages[1] - 1.0).abs() < 1e-10);
        // t=0 (done=false): delta = 1.0 + 0.99*1.0 - 0.5 = 1.49
        // gae = delta (since lambda=0)
        assert!((advantages[0] - 1.49).abs() < 1e-10);
    }
}
