//! Causal identification — determine IF and HOW a causal effect can be estimated.
//!
//! Implements Pearl's backdoor criterion, front-door criterion, d-separation,
//! and instrument discovery.

use std::collections::HashSet;

use crate::graph::CausalGraph;

/// Find the backdoor adjustment set for estimating the causal effect of
/// `treatment` on `outcome`.
///
/// **Backdoor criterion** (Pearl, 2009):
/// A set Z satisfies the backdoor criterion relative to (X, Y) if:
/// 1. No node in Z is a descendant of X
/// 2. Z blocks every back-door path between X and Y
///
/// Returns the set of variables to control for in regression.
///
/// # Example
/// ```
/// use causal_inference::{AdjacencyGraph, find_adjustment_set};
///
/// let g = AdjacencyGraph::from_edges(&[
///     ("Z", "X"), ("Z", "Y"), ("X", "Y"),
/// ]);
/// let adj = find_adjustment_set(&g, "X", "Y");
/// assert!(adj.contains(&"Z".to_string()));
/// ```
pub fn find_adjustment_set(graph: &impl CausalGraph, treatment: &str, outcome: &str) -> Vec<String> {
    let parents = graph.parents(treatment);
    let descendants = graph.descendants(treatment);

    parents.into_iter()
        .filter(|p| {
            !descendants.contains(p)
                && p != treatment
                && p != outcome
        })
        .collect()
}

/// Check if a causal effect is identifiable (there exists a directed path
/// from treatment to outcome).
pub fn is_identifiable(graph: &impl CausalGraph, treatment: &str, outcome: &str) -> bool {
    let descendants = graph.descendants(treatment);
    descendants.contains(outcome)
}

/// Find a front-door adjustment set for estimating X → Y.
///
/// **Front-door criterion** (Pearl, 2009):
/// A set M satisfies the front-door criterion relative to (X, Y) if:
/// 1. X intercepts all directed paths from X to M
/// 2. No unblocked back-door path from X to M
/// 3. All back-door paths from M to Y are blocked by X
///
/// In practice: M is a set of mediators where X → M → Y with no direct X → Y edge.
///
/// Returns `None` if no front-door set exists.
pub fn find_frontdoor_set(graph: &impl CausalGraph, treatment: &str, outcome: &str) -> Option<Vec<String>> {
    let treatment_children = graph.children(treatment);
    let outcome_ancestors = graph.ancestors(outcome);

    // Direct edge X→Y means front-door doesn't work
    if graph.has_edge(treatment, outcome) {
        return None;
    }

    let mediators: Vec<String> = treatment_children
        .into_iter()
        .filter(|child| child != outcome && outcome_ancestors.contains(child))
        .collect();

    if mediators.is_empty() {
        None
    } else {
        Some(mediators)
    }
}

/// Find potential instrumental variables for a treatment→outcome pair.
///
/// An instrument Z must satisfy:
/// 1. **Relevance**: Z causes X (is a parent of treatment)
/// 2. **Exclusion**: Z does not directly cause Y
/// 3. **Independence**: Z is not confounded with Y
pub fn find_instruments(graph: &impl CausalGraph, treatment: &str, outcome: &str) -> Vec<String> {
    let treatment_causes = graph.parents(treatment);
    let outcome_causes: HashSet<String> = graph.parents(outcome).into_iter().collect();
    let outcome_descendants = graph.descendants(outcome);

    treatment_causes
        .into_iter()
        .filter(|z| {
            z != treatment
                && z != outcome
                && !outcome_causes.contains(z)
                && !outcome_descendants.contains(z)
                && !graph.has_edge(z, outcome) // Exclusion restriction
        })
        .collect()
}

/// Test if X and Y are d-separated given conditioning set Z.
///
/// Uses the **Bayes-Ball algorithm** (Shachter, 1998):
/// X ⊥ Y | Z iff no active path exists from X to Y given Z.
///
/// Active path rules:
/// - **Chain** (A→B→C): B not in Z → active
/// - **Fork** (A←B→C): B not in Z → active
/// - **Collider** (A→B←C): B or descendant of B in Z → active
///
/// # Example
/// ```
/// use causal_inference::{AdjacencyGraph, d_separated};
/// use std::collections::HashSet;
///
/// let g = AdjacencyGraph::from_edges(&[
///     ("Z", "X"), ("Z", "Y"), ("X", "Y"),
/// ]);
///
/// // X and Y are NOT d-separated given empty set (back-door through Z)
/// assert!(!d_separated(&g, "X", "Y", &HashSet::new()));
///
/// // X and Y ARE d-separated given {Z} (confounder blocked)
/// // (Not actually true here because X→Y is a direct edge)
/// ```
pub fn d_separated(
    graph: &impl CausalGraph,
    x: &str,
    y: &str,
    z: &HashSet<String>,
) -> bool {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum Dir { Up, Down }

    // Find all ancestors of Z (for collider activation)
    let z_ancestors: HashSet<String> = {
        let mut ancestors = z.clone();
        let mut queue: Vec<String> = z.iter().cloned().collect();
        while let Some(node) = queue.pop() {
            for parent in graph.parents(&node) {
                if ancestors.insert(parent.clone()) {
                    queue.push(parent);
                }
            }
        }
        ancestors
    };

    let mut visited: HashSet<(String, Dir)> = HashSet::new();
    let mut queue: Vec<(String, Dir)> = vec![
        (x.to_string(), Dir::Up),
        (x.to_string(), Dir::Down),
    ];

    while let Some((current, dir)) = queue.pop() {
        if !visited.insert((current.clone(), dir)) {
            continue;
        }
        if current == y {
            return false; // Found active path → NOT d-separated
        }

        let in_z = z.contains(&current);

        match dir {
            Dir::Up => {
                if !in_z {
                    // Chain/Fork: pass through to parents and children
                    for parent in graph.parents(&current) {
                        queue.push((parent, Dir::Up));
                    }
                    for child in graph.children(&current) {
                        queue.push((child, Dir::Down));
                    }
                }
            }
            Dir::Down => {
                if !in_z {
                    // Chain: pass through to children
                    for child in graph.children(&current) {
                        queue.push((child, Dir::Down));
                    }
                }
                // Collider: if conditioned (or ancestor of conditioned), activate
                if z_ancestors.contains(&current) {
                    for parent in graph.parents(&current) {
                        queue.push((parent, Dir::Up));
                    }
                }
            }
        }
    }

    true // No active path found → d-separated
}

/// Find the Markov blanket of a node: parents + children + co-parents.
///
/// Knowing the Markov blanket renders the node conditionally independent
/// of all other nodes in the graph.
pub fn markov_blanket(graph: &impl CausalGraph, node: &str) -> HashSet<String> {
    let mut blanket = HashSet::new();

    for parent in graph.parents(node) {
        blanket.insert(parent);
    }

    for child in graph.children(node) {
        blanket.insert(child.clone());
        // Co-parents of each child
        for co_parent in graph.parents(&child) {
            if co_parent != node {
                blanket.insert(co_parent);
            }
        }
    }

    blanket
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjacencyGraph;

    fn confounded_graph() -> AdjacencyGraph {
        // Z → X, Z → Y, X → Y
        AdjacencyGraph::from_edges(&[
            ("Z", "X"),
            ("Z", "Y"),
            ("X", "Y"),
        ])
    }

    fn chain_graph() -> AdjacencyGraph {
        // X → M → Y  (with unobserved U → X, U → Y)
        AdjacencyGraph::from_edges(&[
            ("X", "M"),
            ("M", "Y"),
            ("U", "X"),
            ("U", "Y"),
        ])
    }

    fn iv_graph() -> AdjacencyGraph {
        // Z → X → Y, U → X, U → Y
        AdjacencyGraph::from_edges(&[
            ("Z", "X"),
            ("X", "Y"),
            ("U", "X"),
            ("U", "Y"),
        ])
    }

    #[test]
    fn test_backdoor() {
        let g = confounded_graph();
        let adj = find_adjustment_set(&g, "X", "Y");
        assert!(adj.contains(&"Z".to_string()));
    }

    #[test]
    fn test_identifiable() {
        let g = confounded_graph();
        assert!(is_identifiable(&g, "X", "Y"));
        assert!(!is_identifiable(&g, "Y", "X")); // No path Y→X
    }

    #[test]
    fn test_frontdoor() {
        let g = chain_graph();
        let fd = find_frontdoor_set(&g, "X", "Y");
        assert!(fd.is_some());
        assert!(fd.unwrap().contains(&"M".to_string()));
    }

    #[test]
    fn test_instruments() {
        let g = iv_graph();
        let ivs = find_instruments(&g, "X", "Y");
        assert!(ivs.contains(&"Z".to_string()));
        assert!(!ivs.contains(&"U".to_string()));
    }

    #[test]
    fn test_d_separation() {
        let g = confounded_graph();
        let empty = HashSet::new();
        // X and Y are NOT d-separated given {} (path through Z)
        assert!(!d_separated(&g, "X", "Y", &empty));

        // X and Y are still NOT d-separated given {Z} because X→Y exists
        let z_set: HashSet<String> = ["Z".to_string()].into();
        assert!(!d_separated(&g, "X", "Y", &z_set));
    }

    #[test]
    fn test_d_separation_collider() {
        // X → C ← Y (collider)
        let g = AdjacencyGraph::from_edges(&[
            ("X", "C"),
            ("Y", "C"),
        ]);
        let empty = HashSet::new();
        // X and Y ARE d-separated given {} (collider blocks)
        assert!(d_separated(&g, "X", "Y", &empty));

        // X and Y are NOT d-separated given {C} (conditioning on collider opens path)
        let c_set: HashSet<String> = ["C".to_string()].into();
        assert!(!d_separated(&g, "X", "Y", &c_set));
    }

    #[test]
    fn test_markov_blanket() {
        let g = confounded_graph();
        let mb = markov_blanket(&g, "X");
        assert!(mb.contains("Z")); // parent
        assert!(mb.contains("Y")); // child
        // Z is also co-parent of Y (Z→Y and X→Y)
    }
}
