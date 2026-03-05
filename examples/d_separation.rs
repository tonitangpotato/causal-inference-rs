//! D-separation and graph theory example.
//!
//! Demonstrates conditional independence testing on a causal DAG.
//!
//! Run: `cargo run --example d_separation`

use std::collections::HashSet;
use causal_inference::{AdjacencyGraph, d_separated, markov_blanket};

fn main() {
    // A classic "Berkson's paradox" structure:
    //
    //   Talent ──→ Hollywood ←── Beauty
    //                  │
    //                  ↓
    //              Fame
    //
    let graph = AdjacencyGraph::from_edges(&[
        ("talent", "hollywood"),
        ("beauty", "hollywood"),
        ("hollywood", "fame"),
    ]);

    println!("=== D-Separation Tests ===\n");

    // Talent and Beauty are independent (d-separated)
    let empty: HashSet<String> = HashSet::new();
    let sep = d_separated(&graph, "talent", "beauty", &empty);
    println!("talent ⊥ beauty | {{}}:             {} (collider blocks)", sep);
    assert!(sep);

    // But conditioning on the collider (hollywood) opens the path!
    // This is Berkson's paradox: among actors, talent and beauty become negatively correlated
    let cond_hollywood: HashSet<String> = ["hollywood".to_string()].into();
    let sep2 = d_separated(&graph, "talent", "beauty", &cond_hollywood);
    println!("talent ⊥ beauty | {{hollywood}}:    {} (collider opened!)", sep2);
    assert!(!sep2);

    // Conditioning on a descendant of the collider also opens it
    let cond_fame: HashSet<String> = ["fame".to_string()].into();
    let sep3 = d_separated(&graph, "talent", "beauty", &cond_fame);
    println!("talent ⊥ beauty | {{fame}}:         {} (descendant of collider)", sep3);
    assert!(!sep3);

    println!("\n=== Markov Blankets ===\n");

    let mb = markov_blanket(&graph, "hollywood");
    println!("Markov blanket of 'hollywood': {:?}", mb);
    println!("  (parents: talent, beauty; child: fame)");

    let mb2 = markov_blanket(&graph, "talent");
    println!("Markov blanket of 'talent':    {:?}", mb2);
    println!("  (child: hollywood; co-parent: beauty)");

    println!("\nBerkson's paradox: In the general population, talent and beauty are");
    println!("independent. But if you only look at Hollywood actors (condition on the");
    println!("collider), they appear negatively correlated — the less talented actors");
    println!("in Hollywood tend to be more beautiful, because they needed beauty to");
    println!("compensate for less talent.");
}
