# Changelog

## 0.1.0 (2025-03-05)

Initial release — the first Rust crate for Pearl's causal inference hierarchy.

### Features

- **Graph**: `CausalGraph` trait (implement for your own graph type) + `AdjacencyGraph` built-in
- **Identification**: Backdoor criterion, front-door criterion, d-separation (Bayes-Ball), Markov blankets, instrumental variable discovery
- **Estimation**: OLS with backdoor adjustment, 2SLS instrumental variables, front-door adjustment, counterfactual queries
- **Refutation**: Placebo treatment, random common cause, data subset tests
- **Time Series**: Granger causality with F-test
- **Statistics**: OLS regression, t-tests, F-tests, p-values via `nalgebra` + `statrs`

### Origin

Extracted from a production causal intelligence system ([The Unusual](https://the-unusual.vercel.app)) that monitors 84 causal nodes across 7 domains (geopolitical, macro, energy, tech, international, China, financial) with 177 directed edges.
