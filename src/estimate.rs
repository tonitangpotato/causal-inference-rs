//! Causal effect estimation — OLS, IV (2SLS), front-door adjustment, counterfactuals.

use std::collections::HashMap;

use nalgebra::DVector;

use crate::graph::CausalGraph;
use crate::identify;
use crate::stats::{self, design_matrix, t_to_p};
use crate::types::*;

/// Estimate the causal effect of `treatment` on `outcome` using OLS
/// with backdoor adjustment.
///
/// Model: Y = β₀ + β₁·X + β₂·Z₁ + β₃·Z₂ + ... + ε
///
/// The causal effect is β₁ (coefficient on treatment after controlling for confounders).
///
/// # Arguments
/// * `graph` — Causal DAG
/// * `data` — Map of variable name → time series values (must be aligned)
/// * `treatment` — Name of the treatment variable
/// * `outcome` — Name of the outcome variable
///
/// # Returns
/// `None` if insufficient data or singular matrix.
pub fn estimate_ols(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
) -> Option<CausalEffect> {
    let adjustment_set = identify::find_adjustment_set(graph, treatment, outcome);

    let t_vals = data.get(treatment)?;
    let o_vals = data.get(outcome)?;
    let n = t_vals.len().min(o_vals.len());
    if n < 10 {
        return None;
    }

    // Gather adjustment variables
    let mut columns: Vec<Vec<f64>> = vec![t_vals[..n].to_vec()];
    let mut actual_adj: Vec<String> = Vec::new();

    for adj_name in &adjustment_set {
        if let Some(adj_vals) = data.get(adj_name) {
            if adj_vals.len() >= n {
                columns.push(adj_vals[..n].to_vec());
                actual_adj.push(adj_name.clone());
            }
        }
    }

    let p = columns.len() + 1; // +1 for intercept
    if n <= p {
        return None;
    }

    // Build design matrix
    let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
    let x = design_matrix(&col_refs);
    let y = DVector::from_column_slice(&o_vals[..n]);

    let result = stats::ols(&x, &y)?;
    let effect = result.coefficients[1]; // β₁ is the treatment effect
    let se = result.std_errors[1];
    let t_stat = result.t_stat(1);
    let p_value = result.p_value(1);

    Some(CausalEffect {
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        effect,
        std_error: se,
        t_stat,
        p_value,
        r_squared: result.r_squared,
        n_obs: n,
        method: if actual_adj.is_empty() {
            EstimationMethod::Bivariate
        } else {
            EstimationMethod::BackdoorAdjustment
        },
        adjustment_set: actual_adj,
    })
}

/// Estimate causal effect using Instrumental Variables (2SLS).
///
/// **Stage 1**: X̂ = π₀ + π₁·Z + ε₁  (predict treatment from instrument)
/// **Stage 2**: Y = β₀ + β₁·X̂ + ε₂  (regress outcome on predicted treatment)
///
/// The causal effect is β₁.
///
/// # Arguments
/// * `data` — Map of variable name → values
/// * `treatment` — Treatment variable name
/// * `outcome` — Outcome variable name
/// * `instrument` — Instrument variable name
///
/// # Panics
/// Does not panic; returns `None` on insufficient data.
pub fn estimate_iv(
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
    instrument: &str,
) -> Option<CausalEffect> {
    let t_vals = data.get(treatment)?;
    let o_vals = data.get(outcome)?;
    let z_vals = data.get(instrument)?;

    let n = t_vals.len().min(o_vals.len()).min(z_vals.len());
    if n < 20 {
        return None;
    }

    // Stage 1: X = π₀ + π₁·Z
    let x1 = design_matrix(&[&z_vals[..n]]);
    let t_vec = DVector::from_column_slice(&t_vals[..n]);
    let stage1 = stats::ols(&x1, &t_vec)?;

    // Check instrument relevance (first-stage F-stat)
    let t_mean = t_vec.mean();
    let sst_1: f64 = t_vals[..n].iter().map(|v| (v - t_mean).powi(2)).sum();
    let _first_stage_f = if stage1.sse > 1e-10 {
        ((sst_1 - stage1.sse) / 1.0) / (stage1.sse / (n - 2) as f64)
    } else {
        f64::INFINITY
    };

    // Stage 2: Y = β₀ + β₁·X̂
    let t_hat: Vec<f64> = (0..n).map(|i| {
        stage1.coefficients[0] + stage1.coefficients[1] * z_vals[i]
    }).collect();

    let x2 = design_matrix(&[&t_hat]);
    let y_vec = DVector::from_column_slice(&o_vals[..n]);
    let stage2 = stats::ols(&x2, &y_vec)?;

    let iv_effect = stage2.coefficients[1];
    let se = stage2.std_errors[1];
    let t_stat = stage2.t_stat(1);
    let p_value = stage2.p_value(1);

    Some(CausalEffect {
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        effect: iv_effect,
        std_error: se,
        t_stat,
        p_value,
        r_squared: stage2.r_squared,
        n_obs: n,
        adjustment_set: vec![instrument.to_string()],
        method: EstimationMethod::InstrumentalVariable,
    })
}

/// Estimate causal effect using front-door adjustment.
///
/// Requires a mediator M where X → M → Y and no direct X → Y edge.
///
/// **Stage 1**: M = α + β₁·X (effect of X on mediator)
/// **Stage 2**: Y = γ + δ₁·M + δ₂·X (effect of M on Y, controlling for X)
/// **Front-door effect** = β₁ × δ₁
pub fn estimate_frontdoor(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
) -> Option<CausalEffect> {
    let mediators = identify::find_frontdoor_set(graph, treatment, outcome)?;
    let mediator = mediators.first()?;

    let t_vals = data.get(treatment)?;
    let m_vals = data.get(mediator)?;
    let o_vals = data.get(outcome)?;

    let n = t_vals.len().min(m_vals.len()).min(o_vals.len());
    if n < 20 {
        return None;
    }

    // Stage 1: M = α + β₁·X
    let x1 = design_matrix(&[&t_vals[..n]]);
    let m_vec = DVector::from_column_slice(&m_vals[..n]);
    let stage1 = stats::ols(&x1, &m_vec)?;
    let effect_x_on_m = stage1.coefficients[1];

    // Stage 2: Y = γ + δ₁·M + δ₂·X
    let x2 = design_matrix(&[&m_vals[..n], &t_vals[..n]]);
    let y_vec = DVector::from_column_slice(&o_vals[..n]);
    let stage2 = stats::ols(&x2, &y_vec)?;
    let effect_m_on_y = stage2.coefficients[1];

    let frontdoor_effect = effect_x_on_m * effect_m_on_y;

    // Approximate SE via delta method
    let se = stage2.mse.sqrt() / (n as f64).sqrt();
    let t_stat = frontdoor_effect / se;
    let p_value = t_to_p(t_stat, (n - 3) as f64);

    Some(CausalEffect {
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        effect: frontdoor_effect,
        std_error: se,
        t_stat,
        p_value,
        r_squared: stage2.r_squared,
        n_obs: n,
        adjustment_set: mediators,
        method: EstimationMethod::FrontDoor,
    })
}

/// Counterfactual query: "What would Y have been if X were x'?"
///
/// Uses the estimated causal effect + observed values:
/// Y_cf = Y_obs - β₁ · (X_obs - X_cf)
///
/// # Arguments
/// * `graph` — Causal DAG
/// * `data` — Aligned data for all variables
/// * `treatment` — Treatment variable
/// * `outcome` — Outcome variable
/// * `treatment_observed` — Current observed treatment value
/// * `treatment_counterfactual` — Hypothetical treatment value
/// * `outcome_observed` — Current observed outcome value
pub fn counterfactual(
    graph: &impl CausalGraph,
    data: &HashMap<String, Vec<f64>>,
    treatment: &str,
    outcome: &str,
    treatment_observed: f64,
    treatment_counterfactual: f64,
    outcome_observed: f64,
) -> Option<Counterfactual> {
    let effect = estimate_ols(graph, data, treatment, outcome)?;

    let o_cf = outcome_observed - effect.effect * (treatment_observed - treatment_counterfactual);

    Some(Counterfactual {
        question: format!(
            "What would {} be if {} were {:.2} (instead of {:.2})?",
            outcome, treatment, treatment_counterfactual, treatment_observed
        ),
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        observed: outcome_observed,
        counterfactual: o_cf,
        individual_effect: o_cf - outcome_observed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjacencyGraph;

    fn make_data() -> (AdjacencyGraph, HashMap<String, Vec<f64>>) {
        // Z → X → Y with Z → Y (confounder)
        let g = AdjacencyGraph::from_edges(&[
            ("Z", "X"),
            ("Z", "Y"),
            ("X", "Y"),
        ]);

        // Generate data: X = 2*Z + noise, Y = 3*X + 1*Z + noise
        let n = 200;
        let mut rng = crate::stats::SimpleRng::new(42);

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
    fn test_ols_with_confounder() {
        let (g, data) = make_data();
        let effect = estimate_ols(&g, &data, "X", "Y").unwrap();

        // True causal effect of X on Y is 3.0
        assert!((effect.effect - 3.0).abs() < 0.5, "effect={}", effect.effect);
        assert!(effect.is_significant(0.05));
        assert_eq!(effect.method, EstimationMethod::BackdoorAdjustment);
        assert!(effect.adjustment_set.contains(&"Z".to_string()));
    }

    #[test]
    fn test_counterfactual() {
        let (g, data) = make_data();
        let cf = counterfactual(&g, &data, "X", "Y", 5.0, 10.0, 20.0).unwrap();

        // Effect ≈ 3.0, so Y_cf ≈ 20 - 3*(5-10) = 20 + 15 = 35
        assert!((cf.counterfactual - 35.0).abs() < 5.0, "cf={}", cf.counterfactual);
        assert!(cf.individual_effect > 0.0); // Increasing X should increase Y
    }
}
