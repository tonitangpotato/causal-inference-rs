//! Causal graph trait and built-in adjacency list implementation.
//!
//! The [`CausalGraph`] trait defines the minimal interface needed for causal inference.
//! Implement it for your own graph type, or use the provided [`AdjacencyGraph`].

use std::collections::{HashMap, HashSet};

/// Trait for a directed acyclic graph (DAG) used in causal inference.
///
/// Implement this trait for your own graph type to use all identification
/// and estimation functions in this crate.
///
/// # Required Methods
///
/// - `parents(node)` — direct causes of a node
/// - `children(node)` — direct effects of a node  
/// - `nodes()` — all node names
/// - `has_node(node)` — check if node exists
/// - `has_edge(from, to)` — check if directed edge exists
pub trait CausalGraph {
    /// Get the direct parents (causes) of a node.
    fn parents(&self, node: &str) -> Vec<String>;

    /// Get the direct children (effects) of a node.
    fn children(&self, node: &str) -> Vec<String>;

    /// Get all node names in the graph.
    fn nodes(&self) -> Vec<String>;

    /// Check if a node exists.
    fn has_node(&self, node: &str) -> bool;

    /// Check if a directed edge exists from `from` to `to`.
    fn has_edge(&self, from: &str, to: &str) -> bool;

    /// Get all ancestors of a node (transitive parents).
    fn ancestors(&self, node: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue = vec![node.to_string()];
        while let Some(current) = queue.pop() {
            for parent in self.parents(&current) {
                if result.insert(parent.clone()) {
                    queue.push(parent);
                }
            }
        }
        result
    }

    /// Get all descendants of a node (transitive children).
    fn descendants(&self, node: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue = vec![node.to_string()];
        while let Some(current) = queue.pop() {
            for child in self.children(&current) {
                if result.insert(child.clone()) {
                    queue.push(child);
                }
            }
        }
        result
    }

    /// Number of nodes.
    fn node_count(&self) -> usize {
        self.nodes().len()
    }
}

/// Simple adjacency-list based causal graph.
///
/// Good for testing and small-to-medium graphs. For large graphs,
/// implement [`CausalGraph`] on your own petgraph/custom structure.
///
/// # Example
/// ```
/// use causal_inference::AdjacencyGraph;
///
/// let mut g = AdjacencyGraph::new();
/// g.add_node("X");
/// g.add_node("Y");
/// g.add_node("Z");
/// g.add_edge("X", "Y"); // X causes Y
/// g.add_edge("Z", "X"); // Z confounds X
/// g.add_edge("Z", "Y"); // Z confounds Y
/// ```
#[derive(Debug, Clone, Default)]
pub struct AdjacencyGraph {
    children_map: HashMap<String, Vec<String>>,
    parents_map: HashMap<String, Vec<String>>,
    node_set: HashSet<String>,
}

impl AdjacencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, name: &str) {
        self.node_set.insert(name.to_string());
        self.children_map.entry(name.to_string()).or_default();
        self.parents_map.entry(name.to_string()).or_default();
    }

    /// Add a directed edge from `from` to `to`.
    /// Both nodes are auto-created if they don't exist.
    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.add_node(from);
        self.add_node(to);
        self.children_map.entry(from.to_string()).or_default().push(to.to_string());
        self.parents_map.entry(to.to_string()).or_default().push(from.to_string());
    }

    /// Build a graph from a list of edges: `[(from, to), ...]`
    pub fn from_edges(edges: &[(&str, &str)]) -> Self {
        let mut g = Self::new();
        for (from, to) in edges {
            g.add_edge(from, to);
        }
        g
    }
}

impl CausalGraph for AdjacencyGraph {
    fn parents(&self, node: &str) -> Vec<String> {
        self.parents_map.get(node).cloned().unwrap_or_default()
    }

    fn children(&self, node: &str) -> Vec<String> {
        self.children_map.get(node).cloned().unwrap_or_default()
    }

    fn nodes(&self) -> Vec<String> {
        self.node_set.iter().cloned().collect()
    }

    fn has_node(&self, node: &str) -> bool {
        self.node_set.contains(node)
    }

    fn has_edge(&self, from: &str, to: &str) -> bool {
        self.children_map
            .get(from)
            .map_or(false, |children| children.contains(&to.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_graph() {
        let g = AdjacencyGraph::from_edges(&[
            ("Z", "X"),
            ("Z", "Y"),
            ("X", "Y"),
        ]);
        assert_eq!(g.node_count(), 3);
        assert!(g.has_edge("Z", "X"));
        assert!(!g.has_edge("X", "Z"));
        assert_eq!(g.parents("Y"), vec!["Z".to_string(), "X".to_string()]);
        assert_eq!(g.children("Z"), vec!["X".to_string(), "Y".to_string()]);
    }

    #[test]
    fn test_ancestors_descendants() {
        let g = AdjacencyGraph::from_edges(&[
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
        ]);
        let anc = g.ancestors("D");
        assert!(anc.contains("A"));
        assert!(anc.contains("B"));
        assert!(anc.contains("C"));
        assert!(!anc.contains("D"));

        let desc = g.descendants("A");
        assert!(desc.contains("B"));
        assert!(desc.contains("C"));
        assert!(desc.contains("D"));
    }
}
