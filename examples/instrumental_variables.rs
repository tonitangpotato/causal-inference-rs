//! Instrumental variable estimation (2SLS) example.
//!
//! Scenario: Estimating the effect of education on wages,
//! using distance to college as an instrument.
//!
//! Run: `cargo run --example instrumental_variables`

use std::collections::HashMap;
use causal_inference::{AdjacencyGraph, find_instruments, estimate_iv};

fn main() {
    // Causal graph:
    //   distance → education → wages
    //   ability → education
    //   ability → wages
    //
    // "ability" is unobserved, confounding education→wages.
    // "distance" is a valid instrument (affects education, not wages directly).
    let graph = AdjacencyGraph::from_edges(&[
        ("distance", "education"),
        ("education", "wages"),
        ("ability", "education"),
        ("ability", "wages"),
    ]);

    // Discover valid instruments
    let instruments = find_instruments(&graph, "education", "wages");
    println!("Valid instruments for education→wages: {:?}", instruments);

    // Generate data
    // True model: education = -0.3*distance + 2*ability + noise
    //             wages = 5*education + 3*ability + noise
    let n = 1000;
    let mut rng = SimpleRng(42);

    let ability: Vec<f64> = (0..n).map(|_| rng.f64() * 5.0).collect();
    let distance: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0 + 5.0).collect();
    let education: Vec<f64> = ability.iter().zip(distance.iter())
        .map(|(a, d)| 2.0 * a - 0.3 * d + 12.0 + rng.f64()).collect();
    let wages: Vec<f64> = education.iter().zip(ability.iter())
        .map(|(e, a)| 5.0 * e + 3.0 * a + 20.0 + rng.f64() * 2.0).collect();

    let mut data = HashMap::new();
    data.insert("distance".to_string(), distance);
    data.insert("education".to_string(), education);
    data.insert("wages".to_string(), wages);
    data.insert("ability".to_string(), ability);

    // IV estimation using distance as instrument
    let iv_effect = estimate_iv(&data, "education", "wages", "distance")
        .expect("IV estimation failed");

    println!("\nIV (2SLS) estimate:");
    println!("{}", iv_effect);
    println!("  True causal effect of education on wages: 5.0");
    println!("  IV estimate: {:.4}", iv_effect.effect);

    // Compare with naive OLS (which is biased by ability confounding)
    // Naive OLS would overestimate because ability↑ → education↑ AND ability↑ → wages↑
    println!("\nNote: Naive OLS would be biased upward due to ability confounding.");
    println!("IV corrects this by using distance (which only affects wages through education).");
}

struct SimpleRng(u64);
impl SimpleRng {
    fn f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }
}
