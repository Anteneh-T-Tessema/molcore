use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub atomic_num: u8,
    pub is_aromatic: bool,
    pub formal_charge: i8,
    pub num_hs: u8,
    // --- enriched features (assigned by post-processing in ingest) ---
    pub degree: u8,         // heavy-atom neighbor count
    pub in_ring: bool,
    pub hybridization: u8,  // 0=unknown, 1=sp, 2=sp2, 3=sp3
    pub chirality: u8,      // 0=none, 1=@ (anticlockwise/S), 2=@@ (clockwise/R)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bond {
    pub bond_type: BondType,
    pub is_aromatic: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

/// Undirected so `graph.edges(idx)` yields all incident bonds without direction bookkeeping.
pub type MolGraph = StableGraph<Atom, Bond, Undirected>;

/// Immutable molecule graph exposed to Python.
/// The petgraph is locked at construction — no mutation after `from_smiles`.
#[pyclass]
#[derive(Clone)]
pub struct PyMolGraph {
    pub graph: MolGraph,
    pub canonical_smiles: String,
}

#[pymethods]
impl PyMolGraph {
    #[staticmethod]
    pub fn from_smiles(smiles: &str) -> PyResult<Self> {
        crate::ingest::ingest(smiles).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("MolIngestionError: {e}"))
        })
    }

    /// Backward-compatible alias for older Python code.
    ///
    /// The name is historical: Rust ingestion currently uses molcore's built-in
    /// parser unless the unfinished `rdkit-backend` feature is completed.
    #[staticmethod]
    pub fn from_smiles_rdkit(smiles: &str) -> PyResult<Self> {
        Self::from_smiles(smiles)
    }

    pub fn canonical_smiles(&self) -> &str {
        &self.canonical_smiles
    }

    pub fn num_atoms(&self) -> usize {
        self.graph.node_count()
    }

    pub fn num_bonds(&self) -> usize {
        self.graph.edge_count()
    }
}
