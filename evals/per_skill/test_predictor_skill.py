"""
Evals for the property_prediction_ml skill.

Covers: PropertyPredictor fit/predict/score/save/load,
predict_with_uncertainty (MC Dropout), MolTorchDataset, multi-task.
"""
import tempfile
import pathlib
import numpy as np
import pytest

from molcore.io import MolDataset, MolTorchDataset
from molcore.predictor import PropertyPredictor

SMILES  = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "c1ccncc1", "Nc1ccccc1"]
LABELS  = np.array([-0.14, 1.68, -0.17, -0.13, 2.89, 0.65, 0.90], dtype=np.float32)
LABELS2 = np.stack([LABELS, LABELS * 2], axis=1)  # (7, 2) multi-task


def _small_ds(labels=None):
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.labels = labels if labels is not None else LABELS
    return ds


def _trained_pred(n_outputs=1, epochs=10) -> PropertyPredictor:
    ds = _small_ds(LABELS if n_outputs == 1 else LABELS2)
    pred = PropertyPredictor(hidden=16, n_layers=2, epochs=epochs, batch_size=4,
                             n_outputs=n_outputs)
    pred.fit(ds, verbose=False)
    return pred


# ── fit ───────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self):
        ds = _small_ds()
        pred = PropertyPredictor(hidden=16, epochs=5, batch_size=4)
        result = pred.fit(ds, verbose=False)
        assert result is pred

    def test_history_populated(self):
        pred = _trained_pred(epochs=5)
        assert len(pred.history["train"]) == 5

    def test_history_with_val(self):
        ds = _small_ds()
        train_ds, val_ds, _ = ds.scaffold_split(train_frac=0.6, val_frac=0.3)
        pred = PropertyPredictor(hidden=16, epochs=5, batch_size=4)
        pred.fit(train_ds, val_dataset=val_ds if len(val_ds) > 0 else None, verbose=False)
        assert len(pred.history["train"]) == 5

    def test_no_labels_raises(self):
        ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
        pred = PropertyPredictor(hidden=16, epochs=3, batch_size=4)
        with pytest.raises(Exception):
            pred.fit(ds, verbose=False)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_shape(self):
        pred = _trained_pred()
        preds = pred.predict(SMILES)
        assert preds.shape == (len(SMILES),)
        assert preds.dtype == np.float32

    def test_invalid_smiles_yields_nan(self):
        pred = _trained_pred()
        preds = pred.predict(["INVALID", "CCO"])
        assert np.isnan(preds[0])
        assert not np.isnan(preds[1])

    def test_all_invalid_returns_all_nan(self):
        pred = _trained_pred()
        preds = pred.predict(["BAD1", "BAD2"])
        assert np.all(np.isnan(preds))

    def test_predict_before_fit_raises(self):
        pred = PropertyPredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            pred.predict(["CCO"])

    def test_multitask_predict_shape(self):
        pred = _trained_pred(n_outputs=2)
        preds = pred.predict(SMILES)
        assert preds.shape == (len(SMILES), 2)


# ── predict_with_uncertainty ─────────────────────────────────────────────────

class TestMCDropout:
    def test_returns_mean_and_std(self):
        pred = _trained_pred(epochs=15)
        mean, std = pred.predict_with_uncertainty(SMILES, n_samples=10)
        assert mean.shape == (len(SMILES),)
        assert std.shape  == (len(SMILES),)

    def test_std_non_negative(self):
        pred = _trained_pred(epochs=15)
        _, std = pred.predict_with_uncertainty(SMILES, n_samples=10)
        valid = std[~np.isnan(std)]
        assert np.all(valid >= 0)

    def test_invalid_smiles_gives_nan(self):
        pred = _trained_pred(epochs=15)
        mean, std = pred.predict_with_uncertainty(["INVALID", "CCO"], n_samples=5)
        assert np.isnan(mean[0]) and np.isnan(std[0])
        assert not np.isnan(mean[1])

    def test_before_fit_raises(self):
        pred = PropertyPredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            pred.predict_with_uncertainty(["CCO"])


# ── score ─────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_returns_metrics(self):
        pred = _trained_pred(epochs=30)
        ds = _small_ds()
        metrics = pred.score(ds)
        assert set(metrics.keys()) >= {"r2", "mae", "rmse", "n"}

    def test_score_n_equals_valid_mols(self):
        pred = _trained_pred(epochs=10)
        ds = _small_ds()
        metrics = pred.score(ds)
        assert metrics["n"] <= len(SMILES)

    def test_mae_non_negative(self):
        pred = _trained_pred(epochs=10)
        metrics = pred.score(_small_ds())
        assert metrics["mae"] >= 0


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_round_trip(self):
        pred = _trained_pred(epochs=10)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "model.pt"
            pred.save(path)
            pred2 = PropertyPredictor.load(path)
            preds = pred2.predict(SMILES)
            assert preds.shape == (len(SMILES),)
            assert not np.all(np.isnan(preds))

    def test_hparams_preserved(self):
        pred = PropertyPredictor(hidden=32, n_layers=2, dropout=0.2, n_outputs=1)
        ds = _small_ds()
        pred.fit(ds, verbose=False)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "m.pt"
            pred.save(path)
            pred2 = PropertyPredictor.load(path)
        assert pred2.hidden == 32
        assert pred2.n_layers == 2

    def test_save_before_fit_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(RuntimeError, match="No trained model"):
                PropertyPredictor().save(pathlib.Path(td) / "m.pt")


# ── MolTorchDataset ───────────────────────────────────────────────────────────

class TestMolTorchDataset:
    def test_len(self):
        ds = _small_ds()
        tds = MolTorchDataset(ds)
        assert len(tds) == len(SMILES)

    def test_getitem_returns_data(self):
        ds = _small_ds()
        tds = MolTorchDataset(ds)
        item = tds[0]
        assert hasattr(item, "x") and hasattr(item, "edge_index")

    def test_getitem_has_y(self):
        import torch
        ds = _small_ds()
        tds = MolTorchDataset(ds)
        item = tds[0]
        assert item.y is not None
        assert item.y.dtype == torch.float32

    def test_getitem_has_smiles(self):
        ds = _small_ds()
        tds = MolTorchDataset(ds)
        item = tds[0]
        assert hasattr(item, "smiles") and isinstance(item.smiles, str)
