"""Tests for molcore.dti — DTIDataset and DTIPredictor."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from molcore.dti import DTIDataset, DTIPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMILES = ["CC(=O)O", "c1ccccc1", "CCO", "CN1CCC[C@H]1c1cccnc1"]
SEQS   = ["MKTLLIL", "ACDEFGHI", "MKTLLIL", "ACDEFGHI"]
LABELS = [6.5, 7.2, 5.8, 8.1]


# ---------------------------------------------------------------------------
# DTIDataset
# ---------------------------------------------------------------------------

class TestDTIDataset:
    def test_basic_construction(self):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS, labels=LABELS)
        assert len(ds) == 4
        assert ds.labels == LABELS

    def test_no_labels(self):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS)
        assert ds.labels is None
        assert len(ds) == 4

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            DTIDataset(smiles=SMILES[:2], sequences=SEQS)

    def test_labels_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels length"):
            DTIDataset(smiles=SMILES, sequences=SEQS, labels=LABELS[:2])

    def test_scaffold_split_sizes(self):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS, labels=LABELS)
        train, val, test = ds.scaffold_split(train_frac=0.5, val_frac=0.25, seed=0)
        assert len(train) + len(val) + len(test) == len(ds)

    def test_scaffold_split_no_label_leak(self):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS, labels=LABELS)
        train, val, test = ds.scaffold_split(seed=42)
        # every split should be a proper DTIDataset
        for split in (train, val, test):
            assert isinstance(split, DTIDataset)


# ---------------------------------------------------------------------------
# DTIPredictor
# ---------------------------------------------------------------------------

class TestDTIPredictor:
    @pytest.fixture
    def tiny_ds(self):
        return DTIDataset(smiles=SMILES, sequences=SEQS, labels=LABELS)

    @pytest.fixture
    def trained(self, tiny_ds):
        pred = DTIPredictor(hidden=16, n_layers=1, epochs=3, device="cpu")
        pred.fit(tiny_ds, verbose=False)
        return pred

    # construction
    def test_defaults(self):
        pred = DTIPredictor()
        assert pred.hidden == 64
        assert pred.model_type == "gcn"
        assert pred._model is None

    def test_custom_hparams(self):
        pred = DTIPredictor(hidden=32, model_type="gin", epochs=5)
        assert pred.hidden == 32
        assert pred.model_type == "gin"

    # fit
    def test_fit_returns_self(self, tiny_ds):
        pred = DTIPredictor(hidden=16, n_layers=1, epochs=2, device="cpu")
        result = pred.fit(tiny_ds, verbose=False)
        assert result is pred

    def test_fit_sets_model(self, tiny_ds):
        pred = DTIPredictor(hidden=16, n_layers=1, epochs=2, device="cpu")
        pred.fit(tiny_ds, verbose=False)
        assert pred._model is not None

    def test_fit_requires_labels(self):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS)
        pred = DTIPredictor(hidden=16, epochs=1, device="cpu")
        with pytest.raises(ValueError, match="labels"):
            pred.fit(ds)

    def test_fit_with_val_set(self, tiny_ds):
        pred = DTIPredictor(hidden=16, n_layers=1, epochs=4, device="cpu")
        pred.fit(tiny_ds, val_dataset=tiny_ds, verbose=False)
        assert "val" in pred._history
        assert len(pred._history["val"]) == 4

    # predict
    def test_predict_shape(self, trained):
        preds = trained.predict(SMILES, SEQS)
        assert preds.shape == (len(SMILES),)
        assert preds.dtype == np.float32

    def test_predict_no_nans_for_valid_inputs(self, trained):
        preds = trained.predict(SMILES, SEQS)
        assert not np.any(np.isnan(preds)), "Expected no NaN for valid inputs"

    def test_predict_nan_for_invalid_smiles(self, trained):
        preds = trained.predict(["INVALID_SMILES"], ["MKTLLIL"])
        assert np.isnan(preds[0])

    def test_predict_before_fit_raises(self):
        pred = DTIPredictor(device="cpu")
        with pytest.raises(RuntimeError, match="fit"):
            pred.predict(SMILES, SEQS)

    def test_predict_length_mismatch_raises(self, trained):
        with pytest.raises(ValueError, match="same length"):
            trained.predict(SMILES[:2], SEQS)

    # score
    def test_score_keys(self, trained, tiny_ds):
        metrics = trained.score(tiny_ds)
        assert set(metrics.keys()) == {"r2", "mae", "rmse", "n"}
        assert metrics["n"] == len(SMILES)

    def test_score_requires_labels(self, trained):
        ds = DTIDataset(smiles=SMILES, sequences=SEQS)
        with pytest.raises(ValueError, match="labels"):
            trained.score(ds)

    # model variants
    @pytest.mark.parametrize("model_type", ["gcn", "gat", "gin"])
    def test_model_variants(self, tiny_ds, model_type):
        pred = DTIPredictor(hidden=16, n_layers=1, epochs=2, model_type=model_type, device="cpu")
        pred.fit(tiny_ds, verbose=False)
        preds = pred.predict(SMILES, SEQS)
        assert preds.shape == (len(SMILES),)

    # save / load
    def test_save_load_roundtrip(self, trained, tmp_path):
        path = tmp_path / "model.pt"
        trained.save(path)
        assert path.exists()

        loaded = DTIPredictor.load(path, device="cpu")
        orig_preds   = trained.predict(SMILES, SEQS)
        loaded_preds = loaded.predict(SMILES, SEQS)
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)

    def test_save_before_fit_raises(self, tmp_path):
        pred = DTIPredictor(device="cpu")
        with pytest.raises(RuntimeError, match="fit"):
            pred.save(tmp_path / "model.pt")

    def test_load_preserves_hparams(self, trained, tmp_path):
        path = tmp_path / "model.pt"
        trained.save(path)
        loaded = DTIPredictor.load(path)
        assert loaded.hidden    == trained.hidden
        assert loaded.n_layers  == trained.n_layers
        assert loaded.model_type == trained.model_type


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_molcore():
    import molcore
    assert hasattr(molcore, "DTIDataset")
    assert hasattr(molcore, "DTIPredictor")
