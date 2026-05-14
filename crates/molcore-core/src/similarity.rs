use ndarray::{aview1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Pairwise Tanimoto similarity matrix.
/// query  : (Q, nbits) uint8 numpy array (1 bit per byte)
/// library: (L, nbits) uint8 numpy array
/// returns: (Q, L) float32 numpy array — zero-copy via IntoPyArray
///
/// Internally packs bits into u64 words and uses hardware popcount,
/// giving ~40–64x fewer inner-loop iterations vs byte-per-bit comparison.
#[pyfunction]
pub fn tanimoto_matrix<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, u8>,
    library: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q = query.as_array();
    let l = library.as_array();
    let (nq, nbits) = (q.nrows(), q.ncols());
    let nl = l.nrows();

    // Pack both matrices into bit-compact u64 representation once (per matrix, not per pair)
    let q_packed: Vec<Vec<u64>> = (0..nq).map(|qi| pack(q.row(qi))).collect();
    let l_packed: Vec<Vec<u64>> = (0..nl).map(|li| pack(l.row(li))).collect();

    // Parallel: each query row is independent
    let rows: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            (0..nl)
                .map(|li| tanimoto_packed(&q_packed[qi], &l_packed[li]))
                .collect()
        })
        .collect();

    let mut result = Array2::<f32>::zeros((nq, nl));
    for (qi, row) in rows.iter().enumerate() {
        result.row_mut(qi).assign(&aview1(row));
    }

    Ok(result.into_pyarray_bound(py))
}

/// Pack a bit array (1 bit per u8 byte) into u64 words.
/// 2048-bit fp → 32 u64 words. 64× fewer words to AND/OR than byte-per-bit.
fn pack(bits: ndarray::ArrayView1<u8>) -> Vec<u64> {
    let nwords = (bits.len() + 63) / 64;
    let mut packed = vec![0u64; nwords];
    for (i, &b) in bits.iter().enumerate() {
        if b != 0 {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    packed
}

/// Tanimoto on bit-packed fingerprints using hardware popcount.
#[inline(always)]
fn tanimoto_packed(a: &[u64], b: &[u64]) -> f32 {
    let mut and_n = 0u32;
    let mut or_n  = 0u32;
    for (&aw, &bw) in a.iter().zip(b.iter()) {
        and_n += (aw & bw).count_ones();
        or_n  += (aw | bw).count_ones();
    }
    if or_n == 0 { 0.0 } else { and_n as f32 / or_n as f32 }
}
