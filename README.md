# causal-inference

**Causal inference in Rust** — the first Rust implementation of Pearl's causal hierarchy.

[![Crates.io](https://img.shields.io/crates/v/causal-inference.svg)](https://crates.io/crates/causal-inference)
[![Docs](https://docs.rs/causal-inference/badge.svg)](https://docs.rs/causal-inference)
[![License](https://img.shields.io/crates/l/causal-inference.svg)](LICENSE)

Born from a production causal intelligence system ([The Unusual](https://the-unusual.vercel.app)), this crate provides tools for identifying, estimating, and validating causal effects from observational data using directed acyclic graphs (DAGs).

## Why?

- **No Rust causal inference library exists** — this is the first
- **Pure Rust** — no Python dependency, no FFI, just `nalgebra` + `statrs`
- **Production-tested** — extracted from a system processing 20K+ data points across 84 causal nodes
- **DoWhy-compatible concepts** — same identify → estimate → refute workflow

## Pearl's Three-Layer Causal Hierarchy

| Layer | Question | This Crate |
|-------|----------|------------|
| 1. Association | "What if I observe X?" | `granger_causality()` |
| 2. Intervention | "What if I do X?" | `estimate_ols()`, `estimate_iv()`, `estimate_frontdoor()` |
| 3. Counterfactual | "What if X had been different?" | `counterfactual()` |

## Quick Start

```rust
use causal_inference::*;
use std::collections::HashMap;

// 1. Define your causal graph
let graph = AdjacencyGraph::from_edges(&[
    ("confounder", "treatment"),
    ("confounder", "outcome"),
    ("treatment", "outcome"),
]);

// 2. Prepare aligned data
let mut data = HashMap::new();
data.insert("treatment".into(), vec![1.0, 2.0, 3.0, /* ... */]);
data.insert("outcome".into(),   vec![2.1, 4.0, 5.9, /* ... */]);
data.insert("confounder".into(), vec![0.5, 1.0, 1.5, /* ... */]);

// 3. Identify (backdoor criterion)
let adjustment = find_adjustment_set(&graph, "treatment", "outcome");
// → ["confounder"]

// 4. Estimate (OLS with backdoor adjustment)
let effect = estimate_ols(&graph, &data, "treatment", "outcome").unwrap();
println!("{}", effect);
// treatment → outcome: β=2.98 [2.71, 3.25] p=0.0000 ✅ R²=0.987 n=200

// 5. Refute (placebo treatment test)
let placebo = refute_placebo(&graph, &data, "treatment", "outcome", 42).unwrap();
println!("{}", placebo);
// Placebo Treatment: ✅ PASSED (original=2.98, refuted=0.03, p=0.89)
```

## Features

### Identification (Graph Theory)
- **`find_adjustment_set()`** — Backdoor criterion (Pearl 2009)
- **`find_frontdoor_set()`** — Front-door criterion
- **`find_instruments()`** — Discover valid instrumental variables
- **`d_separated()`** — Bayes-Ball algorithm for conditional independence
- **`markov_blanket()`** — Find the Markov blanket of any node
- **`is_identifiable()`** — Check if causal effect is estimable

### Estimation
- **`estimate_ols()`** — OLS with backdoor adjustment (controlling for confounders)
- **`estimate_iv()`** — Two-Stage Least Squares (2SLS) for unobserved confounders
- **`estimate_frontdoor()`** — Front-door adjustment via mediators
- **`counterfactual()`** — "What would Y be if X had been different?"

### Refutation (Validation)
- **`refute_placebo()`** — Replace treatment with random noise (effect should vanish)
- **`refute_random_common_cause()`** — Add random confounder (effect should be stable)
- **`refute_data_subset()`** — Re-estimate on subset (effect should be similar)

### Time Series
- **`granger_causality()`** — Granger causality with F-test (does past X predict Y?)

### Bring Your Own Graph
Implement the `CausalGraph` trait for your own graph type:

```rust
use causal_inference::CausalGraph;

impl CausalGraph for MyPetgraphWrapper {
    fn parents(&self, node: &str) -> Vec<String> { /* ... */ }
    fn children(&self, node: &str) -> Vec<String> { /* ... */ }
    fn nodes(&self) -> Vec<String> { /* ... */ }
    fn has_node(&self, node: &str) -> bool { /* ... */ }
    fn has_edge(&self, from: &str, to: &str) -> bool { /* ... */ }
}
```

## Comparison with DoWhy (Python)

| Feature | DoWhy | causal-inference |
|---------|-------|------------------|
| Language | Python | Rust |
| Backdoor criterion | ✅ | ✅ |
| Front-door criterion | ✅ | ✅ |
| IV estimation | ✅ | ✅ (2SLS) |
| d-separation | ✅ | ✅ (Bayes-Ball) |
| Placebo refutation | ✅ | ✅ |
| Random common cause | ✅ | ✅ |
| Data subset | ✅ | ✅ |
| Granger causality | ❌ | ✅ |
| Counterfactual | ✅ | ✅ |
| Custom graphs | NetworkX | Trait-based |
| Speed | ~1x | ~10-50x |
| No Python needed | ❌ | ✅ |

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Shachter, R. (1998). "Bayes-Ball: The Rational Pastime"
- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models"
- Sharma & Kiciman (2020). "DoWhy: An End-to-End Library for Causal Inference"

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
