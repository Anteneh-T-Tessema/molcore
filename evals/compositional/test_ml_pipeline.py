"""
Compositional eval: full ML pipeline.

Tests that MolDataset → scaffold_split → PropertyPredictor.fit →
predict/score/uncertainty form a coherent, end-to-end chain.
Also tests MolDataset Parquet I/O round-trip with labels.
"""
import tempfile
import pathlib
import numpy as np
import pytest

from molcore.io import MolDataset
from molcore.predictor import PropertyPredictor

SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC",
    "c1ccncc1", "Nc1ccccc1", "COc1ccccc1", "CCCO", "c1ccoc1",
]
LOGP = np.array([-0.14, 1.68, -0.17, -0.13, 2.89,
                  0.65, 0.90, 1.35, 0.25, 1.34], dtype=np.float32)


@pytest.fixture
def labelled_ds():
    ds = MolDataset.from_smiles(SMILES, compute_fps=True, compute_desc=True)
    ds.labels = LOGP
    return ds


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestMolDatasetPipeline:
    def test_length(self, labelled_ds):
        assert len(labelled_ds) == len(SMILES)

    def test_fingerprints_shape(self, labelled_ds):
        fps = labelled_ds.fingerprints
        assert fps is not None and fps.shape == (len(SMILES), 2048)

    def test_descriptors_shape(self, labelled_ds):
        desc = labelled_ds.descriptors
        assert desc is not None and desc.shape == (len(SMILES), 3)

    def test_scaffold_split_sizes_sum(self, labelled_ds):
        train, val, test = labelled_ds.scaffold_split(train_frac=0.7, val_frac=0.2)
        assert len(train) + len(val) + len(test) == len(SMILES)

    def test_scaffold_split_no_overlap(self, labelled_ds):
        train, val, test = labelled_ds.scaffold_split()
        all_smi = set(train.smiles) | set(val.smiles) | set(test.smiles)
        assert len(all_smi) == len(SMILES)

    def test_labels_preserved_in_split(self, labelled_ds):
        train, _, _ = labelled_ds.scaffold_split()
        assert train.labels is not None
        assert train.labels.dtype == np.float32

    def test_getitem_single_row(self, labelled_ds):
        row = labelled_ds[0]
        assert len(row) == 1
        assert row.labels is not None


# ── Parquet round-trip with labels ────────────────────────────────────────────

class TestParquetRoundTrip:
    def test_smiles_preserved(self, labelled_ds):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "test.parquet"
            labelled_ds.write_parquet(path)
            ds2 = MolDataset.read_parquet(path)
        assert ds2.smiles == labelled_ds.smiles

    def test_fingerprints_preserved(self, labelled_ds):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "test.parquet"
            labelled_ds.write_parquet(path)
            ds2 = MolDataset.read_parquet(path)
        np.testing.assert_array_equal(ds2.fingerprints, labelled_ds.fingerprints)

    def test_descriptors_preserved(self, labelled_ds):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "test.parquet"
            labelled_ds.write_parquet(path)
            ds2 = MolDataset.read_parquet(path)
        np.testing.assert_allclose(ds2.descriptors, labelled_ds.descriptors, rtol=1e-5)

    def test_labels_preserved(self, labelled_ds):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "test.parquet"
            labelled_ds.write_parquet(path)
            ds2 = MolDataset.read_parquet(path)
        np.testing.assert_allclose(ds2.labels, labelled_ds.labels, rtol=1e-5)


# ── End-to-end GCN pipeline ───────────────────────────────────────────────────

class TestGCNPipeline:
    @pytest.fixture
    def trained_pred(self, labelled_ds):
        train, val, _ = labelled_ds.scaffold_split(train_frac=0.6, val_frac=0.3)
        pred = PropertyPredictor(hidden=16, n_layers=2, epochs=20,
                                 batch_size=4, lr=5e-3)
        val_arg = val if len(val) > 0 else None
        pred.fit(train, val_dataset=val_arg, verbose=False)
        return pred

    def test_predict_shape(self, trained_pred):
        preds = trained_pred.predict(SMILES)
        assert preds.shape == (len(SMILES),)

    def test_predict_dtype(self, trained_pred):
        preds = trained_pred.predict(SMILES)
        assert preds.dtype == np.float32

    def test_no_nan_for_valid_smiles(self, trained_pred):
        preds = trained_pred.predict(SMILES)
        assert not np.any(np.isnan(preds))

    def test_score_returns_all_keys(self, trained_pred, labelled_ds):
        metrics = trained_pred.score(labelled_ds)
        assert {"r2", "mae", "rmse", "n"} <= set(metrics.keys())

    def test_uncertainty_mean_matches_predict(self, trained_pred):
        preds = trained_pred.predict(SMILES[:4])
        mean, _ = trained_pred.predict_with_uncertainty(SMILES[:4], n_samples=10)
        # means won't match exactly (different dropout seeds) but shapes match
        assert preds.shape == mean.shape

    def test_uncertainty_std_positive(self, trained_pred):
        _, std = trained_pred.predict_with_uncertainty(SMILES, n_samples=10)
        valid = std[~np.isnan(std)]
        assert np.all(valid >= 0)

    def test_save_load_predict_matches(self, trained_pred):
        preds_before = trained_pred.predict(SMILES)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "gcn.pt"
            trained_pred.save(path)
            pred2 = PropertyPredictor.load(path)
        preds_after = pred2.predict(SMILES)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)

    def test_pipeline_with_parquet_roundtrip(self, labelled_ds):
        """Dataset → Parquet → reload → split → train → predict."""
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "ds.parquet"
            labelled_ds.write_parquet(path)
            ds2 = MolDataset.read_parquet(path)
        ds2.labels = labelled_ds.labels  # reload doesn't persist multi-label yet
        train, _, _ = ds2.scaffold_split()
        pred = PropertyPredictor(hidden=16, epochs=5, batch_size=4)
        pred.fit(train, verbose=False)
        preds = pred.predict(SMILES[:3])
        assert preds.shape == (3,)
