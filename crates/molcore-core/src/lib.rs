// PyO3's #[pymethods] macro expansion triggers this lint as a false positive.
#![allow(clippy::useless_conversion)]
use pyo3::prelude::*;

pub mod molecule;
pub mod ingest;
pub mod fingerprints;
pub mod graph_arrays;
pub mod similarity;
pub mod descriptors;

pub use molecule::PyMolGraph;

#[pymodule]
fn _molcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<molecule::PyMolGraph>()?;
    m.add_function(wrap_pyfunction!(fingerprints::ecfp4_batch, m)?)?;
    m.add_function(wrap_pyfunction!(graph_arrays::mol_to_graph_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::tanimoto_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(descriptors::calc_descriptors_batch, m)?)?;
    Ok(())
}
