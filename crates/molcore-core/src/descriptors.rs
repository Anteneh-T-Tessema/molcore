use ndarray::{aview1, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Batch molecular descriptors: MW, LogP (approximate), heavy atom count.
/// Returns (N, 3) float32 numpy array — zero-copy via IntoPyArray.
/// For exact RDKit descriptors (TPSA, Crippen LogP) use rdkit_bridge.
#[pyfunction]
pub fn calc_descriptors_batch<'py>(
    py: Python<'py>,
    smiles_list: Vec<String>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows: Vec<[f32; 3]> = smiles_list
        .par_iter()
        .map(|smi| compute_for(smi))
        .collect();

    let n = rows.len();
    let mut result = Array2::<f32>::zeros((n, 3));
    for (i, row) in rows.iter().enumerate() {
        result.row_mut(i).assign(&aview1(row.as_slice()));
    }

    Ok(result.into_pyarray_bound(py))
}

fn compute_for(smiles: &str) -> [f32; 3] {
    let mol = match crate::ingest::ingest(smiles) {
        Ok(m) => m,
        Err(_) => return [0.0; 3],
    };
    let g = &mol.graph;
    let heavy = g.node_count() as f32;
    let mw    = g.node_indices().map(|i| atomic_mass(g[i].atomic_num)).sum::<f32>();
    let logp  = g.node_indices().map(|i| crippen_fragment(g[i].atomic_num)).sum::<f32>();
    [mw, logp, heavy]
}

fn atomic_mass(atomic_num: u8) -> f32 {
    match atomic_num {
        1  =>  1.008,
        5  => 10.81,
        6  => 12.011,
        7  => 14.007,
        8  => 15.999,
        9  => 18.998,
        15 => 30.974,
        16 => 32.06,
        17 => 35.45,
        35 => 79.904,
        53 => 126.90,
        _  => atomic_num as f32 * 2.0,
    }
}

fn crippen_fragment(atomic_num: u8) -> f32 {
    // Simplified Wildman–Crippen contribution per atom type.
    // For production, use rdkit_bridge which calls RDKit's Crippen.
    match atomic_num {
        6  =>  0.35,
        7  => -1.03,
        8  => -0.74,
        9  =>  0.14,
        16 =>  0.03,
        17 =>  0.60,
        35 =>  0.88,
        _  =>  0.0,
    }
}
