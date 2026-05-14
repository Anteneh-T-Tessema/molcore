use ndarray::{aview1, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Batch ECFP4 — takes N SMILES, returns (N × nbits) uint8 numpy array.
/// Rayon parallelises across molecules; IntoPyArray transfers ownership zero-copy.
#[pyfunction]
#[pyo3(signature = (smiles_list, radius=2, nbits=2048))]
pub fn ecfp4_batch<'py>(
    py: Python<'py>,
    smiles_list: Vec<String>,
    radius: usize,
    nbits: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let fps: Vec<Vec<u8>> = smiles_list
        .par_iter()
        .map(|smi| compute_morgan(smi, radius, nbits))
        .collect();

    let n = fps.len();
    let mut result = Array2::<u8>::zeros((n, nbits));
    for (i, fp) in fps.iter().enumerate() {
        result.row_mut(i).assign(&aview1(fp));
    }

    Ok(result.into_pyarray_bound(py))
}

/// Morgan / ECFP fingerprint — pure Rust, chemically correct, deterministic.
/// Bit vectors differ from RDKit (different hash seeds) — use backend="rdkit"
/// when exact parity with legacy trained models is required.
pub fn compute_morgan(smiles: &str, radius: usize, nbits: usize) -> Vec<u8> {
    use std::collections::HashSet;

    let mol = match crate::ingest::ingest(smiles) {
        Ok(m) => m,
        Err(_) => return vec![0u8; nbits],
    };

    let graph = &mol.graph;
    let mut identifiers: Vec<u64> = graph
        .node_indices()
        .map(|idx| {
            let atom = &graph[idx];
            let h = fnv64(atom.atomic_num as u64);
            let h = fnv64_combine(h, atom.formal_charge as u64);
            fnv64_combine(h, atom.num_hs as u64)
        })
        .collect();

    let mut bits: HashSet<u64> = identifiers.iter().copied().collect();

    for _ in 0..radius {
        let mut new_ids = identifiers.clone();
        for (i, idx) in graph.node_indices().enumerate() {
            let mut neighbors: Vec<u64> = graph
                .neighbors(idx)
                .map(|nb| identifiers[nb.index()])
                .collect();
            neighbors.sort_unstable();
            let mut h = identifiers[i];
            for nb_id in neighbors {
                h = fnv64_combine(h, nb_id);
            }
            new_ids[i] = h;
        }
        identifiers = new_ids;
        bits.extend(identifiers.iter().copied());
    }

    let mut fp = vec![0u8; nbits];
    for bit_hash in bits {
        fp[(bit_hash as usize) % nbits] = 1;
    }
    fp
}

#[inline]
fn fnv64(v: u64) -> u64 {
    const PRIME: u64 = 0x00000100000001B3;
    const BASIS: u64 = 0xcbf29ce484222325;
    BASIS.wrapping_mul(PRIME) ^ v
}

#[inline]
fn fnv64_combine(a: u64, b: u64) -> u64 {
    const PRIME: u64 = 0x00000100000001B3;
    a.wrapping_mul(PRIME) ^ b
}
