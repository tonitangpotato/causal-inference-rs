//! # causal-inference
//!
//! Causal inference in Rust — the first Rust implementation of Pearl's causal hierarchy.
//!
//! Born from a production causal intelligence system ([The Unusual](https://the-unusual.vercel.app)),
//! this crate provides tools for identifying, estimating, and validating causal effects
//! from observational data using directed acyclic graphs (DAGs).
//!
//! ## Pearl's Causal Hierarchy
//!
//! - **Layer 1 (Association)**: Granger causality, correlation analysis
//! - **Layer 2 (Intervention)**: Backdoor adjustment, front-door criterion, instrumental variables
//! - **Layer 3 (Counterfactual)**: "What would have happened if...?"
//!
//! ## Quick Start
//!
//! ```rust
//! use causal_inference::{CausalGraph, AdjacencyGraph, find_adjustment_set};
//!
//! // Define your causal graph
//! let mut graph = AdjacencyGraph::new();
//! graph.add_node("treatment");
//! graph.add_node("outcome");
//! graph.add_node("confounder");
//! graph.add_edge("confounder", "treatment");
//! graph.add_edge("confounder", "outcome");
//! graph.add_edge("treatment", "outcome");
//!
//! // Identify adjustment set (backdoor criterion)
//! let adjustment = find_adjustment_set(&graph, "treatment", "outcome");
//! assert!(adjustment.contains(&"confounder".to_string()));
//! ```
//!
//! ## Features
//!
//! - **Identification**: Backdoor criterion, front-door criterion, d-separation (Bayes-Ball)
//! - **Estimation**: OLS regression, 2SLS instrumental variables, front-door adjustment
//! - **Refutation**: Placebo treatment, random common cause, data subset validation
//! - **Time Series**: Granger causality with F-test
//! - **Graph Theory**: d-separation, Markov blankets, instrument discovery
//! - **No runtime dependencies on Python** — pure Rust with nalgebra + statrs

pub mod graph;
pub mod identify;
pub mod estimate;
pub mod refute;
pub mod granger;
pub mod types;
pub mod stats;

pub use graph::{CausalGraph, AdjacencyGraph};
pub use types::{CausalEffect, EstimationMethod, RefutationResult, GrangerResult, EdgeVerdict};
pub use identify::{find_adjustment_set, find_frontdoor_set, find_instruments, d_separated, markov_blanket, is_identifiable};
pub use estimate::{estimate_ols, estimate_iv, estimate_frontdoor, counterfactual};
pub use refute::{refute_placebo, refute_random_common_cause, refute_data_subset};
pub use granger::granger_causality;
