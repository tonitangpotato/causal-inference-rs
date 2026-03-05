//! Refutation tests — validate that estimated causal effects are real, not spurious.
//!
//! Three tests, matching DoWhy's refutation framework:
//! 1. **Placebo treatment**: Replace treatment with random noise → effect should vanish
//! 2. **Random common cause**: Add random confounder → effect should be stable
//! 3. **Data subset**: Re-estimate on subset → effect should be similar

use std::collections::HashMap;

use crate::graph::CausalGraph;
use crate::estimate;
use crate::stats::{self, design_matrix};
use crate::types::RefutationResult;

use nalgebra::DVector;

/// **Placebo treatment test**: Replace treatment with random noise.
///
/// If the causal effect is real, a random variable should NOT predict the outcome.
/// The test passes if the placebo effect is NOT significant (p > 0.05).
///
/// # Arguments
/// * `graph` — Causal DAG
/// * `data` — Aligned data
/// * `treatment` — Treatment variable
/// * `outcome` — Outcome variable
/// * `seed` — Random seed for reproducibility
pub fn refute_placebo(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
    seed: u64,
) -> Option<RefutationResult> {
    let original = estimate::estimate_ols(graph, data, treatment, outcome)?;

    let o_vals = data.get(outcome)?;
    let n = o_vals.len();
    if n < 20 { return None; }

    // Generate random placebo treatment
    let mut rng = stats::SimpleRng::new(seed);
    let placebo: Vec<f64> = rng.next_n(n);

    // Regress outcome on placebo
    let x = design_matrix(&[&placebo]);
    let y = DVector::from_column_slice(o_vals);
    let result = stats::ols(&x, &y)?;

    let placebo_effect = result.coefficients[1];
    let p_value = result.p_value(1);

    // Placebo should NOT be significant
    let passed = p_value > 0.05;

    Some(RefutationResult {
        test_name: "Placebo Treatment".to_string(),
        original_effect: original.effect,
        refuted_effect: placebo_effect,
        passed,
        p_value,
    })
}

/// **Random common cause test**: Add a random variable as confounder.
///
/// If the causal effect is real, adding a random "confounder" shouldn't change
/// the estimated effect significantly (< 10% change).
pub fn refute_random_common_cause(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
    seed: u64,
) -> Option<RefutationResult> {
    let original = estimate::estimate_ols(graph, data, treatment, outcome)?;

    let t_vals = data.get(treatment)?;
    let o_vals = data.get(outcome)?;
    let n = t_vals.len().min(o_vals.len());
    if n < 20 { return None; }

    // Generate random "confounder"
    let mut rng = stats::SimpleRng::new(seed);
    let random_z: Vec<f64> = rng.next_n(n);

    // Regress outcome on treatment + random confounder
    let x = design_matrix(&[&t_vals[..n], &random_z]);
    let y = DVector::from_column_slice(&o_vals[..n]);
    let result = stats::ols(&x, &y)?;

    let adjusted_effect = result.coefficients[1];

    // Effect should remain similar (within 10% of original)
    let change_ratio = if original.effect.abs() > 1e-10 {
        (adjusted_effect / original.effect - 1.0).abs()
    } else {
        (adjusted_effect - original.effect).abs()
    };
    let passed = change_ratio < 0.10;

    Some(RefutationResult {
        test_name: "Random Common Cause".to_string(),
        original_effect: original.effect,
        refuted_effect: adjusted_effect,
        passed,
        p_value: original.p_value,
    })
}

/// **Data subset test**: Re-estimate on a random subset of the data.
///
/// If the causal effect is robust, it should be similar on a subset.
/// The test passes if the subset effect is within 50% of the original.
///
/// # Arguments
/// * `fraction` — Fraction of data to use (e.g., 0.7 for 70%)
pub fn refute_data_subset(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
    fraction: f64,
) -> Option<RefutationResult> {
    let original = estimate::estimate_ols(graph, data, treatment, outcome)?;

    // Create subset data (take first N*fraction observations)
    let n = data.get(treatment)?.len();
    let subset_n = (n as f64 * fraction) as usize;
    if subset_n < 10 { return None; }

    let subset_data: HashMap<String, Vec<f64>> = data.iter()
        .map(|(k, v)| (k.clone(), v[..subset_n.min(v.len())].to_vec()))
        .collect();

    let subset_effect = estimate::estimate_ols(graph, &subset_data, treatment, outcome)?;

    // Effect should be similar (within 50% of original)
    let ratio = if original.effect.abs() > 1e-10 {
        (subset_effect.effect / original.effect).abs()
    } else {
        1.0
    };
    let passed = ratio > 0.5 && ratio < 2.0;

    Some(RefutationResult {
        test_name: format!("Data Subset ({:.0}%)", fraction * 100.0),
        original_effect: original.effect,
        refuted_effect: subset_effect.effect,
        passed,
        p_value: subset_effect.p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjacencyGraph;

    fn make_data() -> (AdjacencyGraph, HashMap<String, Vec<f64>>) {
        let g = AdjacencyGraph::from_edges(&[
            ("Z", "X"), ("Z", "Y"), ("X", "Y"),
        ]);

        let n = 200;
        let mut rng = stats::SimpleRng::new(42);
        let z: Vec<f64> = (0..n).map(|_| rng.next_f64() * 5.0).collect();
        let x: Vec<f64> = z.iter().map(|zi| 2.0 * zi + rng.next_f64() * 0.5).collect();
        let y: Vec<f64> = x.iter().zip(z.iter())
            .map(|(xi, zi)| 3.0 * xi + 1.0 * zi + rng.next_f64() * 0.5)
            .collect();

        let mut data = HashMap::new();
        data.insert("Z".to_string(), z);
        data.insert("X".to_string(), x);
        data.insert("Y".to_string(), y);
        (g, data)
    }

    #[test]
    fn test_placebo() {
        let (g, data) = make_data();
        let result = refute_placebo(&g, &data, "X", "Y", 99999).unwrap();
        assert!(result.passed, "Placebo should pass for real causal effect: {:?}", result);
    }

    #[test]
    fn test_random_common_cause() {
        let (g, data) = make_data();
        let result = refute_random_common_cause(&g, &data, "X", "Y", 42).unwrap();
        assert!(result.passed, "Random common cause should pass");
    }

    #[test]
    fn test_data_subset() {
        let (g, data) = make_data();
        let result = refute_data_subset(&g, &data, "X", "Y", 0.7).unwrap();
        assert!(result.passed, "Data subset should pass");
    }
}
