use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use crate::molecule::{BondType, PyMolGraph};

/// Pure-Rust version for unit tests — no Python runtime required.
pub fn mol_to_graph_arrays_raw(
    mol: &PyMolGraph,
) -> (Array2<f32>, Array2<i64>, Array2<f32>) {
    let graph = &mol.graph;
    let n = graph.node_count();
    let e = graph.edge_count() * 2;

    let mut node_feats = Array2::<f32>::zeros((n, 4));
    for (i, idx) in graph.node_indices().enumerate() {
        let atom = &graph[idx];
        node_feats[[i, 0]] = atom.atomic_num as f32;
        node_feats[[i, 1]] = atom.is_aromatic as u8 as f32;
        node_feats[[i, 2]] = atom.formal_charge as f32;
        node_feats[[i, 3]] = atom.num_hs as f32;
    }

    let mut src_col = Vec::with_capacity(e);
    let mut dst_col = Vec::with_capacity(e);
    let mut edge_feats = Array2::<f32>::zeros((e.max(1), 4));
    let mut eidx = 0usize;

    for edge in graph.edge_indices() {
        let (s, d) = graph.edge_endpoints(edge).unwrap();
        let bond = &graph[edge];
        let oh = bond_one_hot(bond.bond_type);
        for &(a, b) in &[
            (s.index() as i64, d.index() as i64),
            (d.index() as i64, s.index() as i64),
        ] {
            src_col.push(a);
            dst_col.push(b);
            for k in 0..4 { edge_feats[[eidx, k]] = oh[k]; }
            eidx += 1;
        }
    }

    let mut edge_index = Array2::<i64>::zeros((2, e.max(1)));
    for i in 0..e { edge_index[[0, i]] = src_col[i]; edge_index[[1, i]] = dst_col[i]; }

    // Trim to actual edge count (handles zero-bond molecules)
    let edge_index = edge_index.slice(ndarray::s![.., ..e]).to_owned();
    let edge_feats_out = edge_feats.slice(ndarray::s![..e, ..]).to_owned();

    (node_feats, edge_index, edge_feats_out)
}

/// Extract node features, edge_index, and edge features from a MolGraph.
/// Returns three numpy arrays — all zero-copy via IntoPyArray.
///
/// node_feats : (N, 4) float32  — [atomic_num, is_aromatic, formal_charge, num_hs]
/// edge_index : (2, E) int64    — COO format, bidirectional (each bond → 2 entries)
/// edge_feats : (E, 4) float32  — [single, double, triple, aromatic] one-hot
#[pyfunction]
pub fn mol_to_graph_arrays<'py>(
    py: Python<'py>,
    mol: &PyMolGraph,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let (nf, ei, ef) = mol_to_graph_arrays_raw(mol);
    Ok((
        nf.into_pyarray_bound(py),
        ei.into_pyarray_bound(py),
        ef.into_pyarray_bound(py),
    ))
}

fn bond_one_hot(bt: BondType) -> [f32; 4] {
    match bt {
        BondType::Single   => [1., 0., 0., 0.],
        BondType::Double   => [0., 1., 0., 0.],
        BondType::Triple   => [0., 0., 1., 0.],
        BondType::Aromatic => [0., 0., 0., 1.],
    }
}
