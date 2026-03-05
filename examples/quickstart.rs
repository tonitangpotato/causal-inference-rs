//! Quick start: estimate a causal effect with backdoor adjustment.
//!
//! Run: `cargo run --example quickstart`

use std::collections::HashMap;
use causal_inference::{AdjacencyGraph, estimate_ols, find_adjustment_set, refute_placebo, refute_data_subset};

fn main() {
    // 1. Define your causal graph (DAG)
    //
    //    Z ──→ X ──→ Y
    //    └───────────→│
    //
    // Z confounds X→Y. We need to adjust for Z.
    let graph = AdjacencyGraph::from_edges(&[
        ("Z", "X"),
        ("Z", "Y"),
        ("X", "Y"),
    ]);

    // 2. Check what we need to control for
    let adjustment = find_adjustment_set(&graph, "X", "Y");
    println!("Adjustment set: {:?}", adjustment);
    // → ["Z"]

    // 3. Generate some data (in practice, load your own)
    // True model: X = 2*Z + noise, Y = 3*X + Z + noise
    let n = 500;
    let mut rng = SimpleRng(42);
    let z: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x: Vec<f64> = z.iter().map(|zi| 2.0 * zi + rng.f64()).collect();
    let y: Vec<f64> = x.iter().zip(z.iter())
        .map(|(xi, zi)| 3.0 * xi + zi + rng.f64()).collect();

    let mut data = HashMap::new();
    data.insert("Z".to_string(), z);
    data.insert("X".to_string(), x);
    data.insert("Y".to_string(), y);

    // 4. Estimate the causal effect
    let effect = estimate_ols(&graph, &data, "X", "Y")
        .expect("estimation failed");

    println!("\n{}", effect);
    println!("  True causal effect: 3.0");
    println!("  Estimated:          {:.4}", effect.effect);
    println!("  Significant:        {}", effect.is_significant(0.05));

    // 5. Validate with refutation tests
    let placebo = refute_placebo(&graph, &data, "X", "Y", 12345)
        .expect("placebo test failed");
    println!("\n{}", placebo);

    let subset = refute_data_subset(&graph, &data, "X", "Y", 0.7)
        .expect("subset test failed");
    println!("{}", subset);
}

// Minimal RNG for the example
struct SimpleRng(u64);
impl SimpleRng {
    fn f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }
}
