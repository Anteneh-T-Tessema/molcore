"""Tests for PropertyPredictor — train, predict, score, save/load."""
import tempfile
import pathlib
import numpy as np
import pytest

from molcore.predictor import PropertyPredictor
from molcore.io import MolDataset


# 20-molecule mini ESOL subset (SMILES + log solubility)
_SMILES = [
    "CCO", "CCCO", "CCCCO", "CCCCCO", "CC(O)C",
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
    "CC(=O)O", "CCC(=O)O", "c1ccccc1C(=O)O",
    "c1ccncc1", "c1cccnc1", "c1ccoc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC#N", "CCCC#N", "CCO", "CC(C)=O", "C=O",
]
_LABELS = np.array([
    -0.31, -0.83, -0.93, -1.40, -0.48,
    -1.90, -2.27, -2.71,
    -0.17, -0.11, -1.87,
    -0.91, -0.91,  0.11,
    -0.07,
     0.41, -0.56, -0.31,  0.26,  1.09,
], dtype=np.float32)


def _mini_dataset(n: int = 20) -> MolDataset:
    ds = MolDataset(smiles=_SMILES[:n], labels=_LABELS[:n])
    return ds


# ── basic fit/predict ─────────────────────────────────────────────────────────

def test_fit_runs_without_error():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, n_layers=2, epochs=5, batch_size=8)
    pred.fit(ds, verbose=False)
    assert pred._model is not None


def test_predict_returns_array():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, n_layers=2, epochs=5, batch_size=8)
    pred.fit(ds, verbose=False)
    out = pred.predict(_SMILES[:5])
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)
    assert out.dtype == np.float32


def test_predict_before_fit_raises():
    pred = PropertyPredictor()
    with pytest.raises(RuntimeError, match="fit"):
        pred.predict(["CCO"])


def test_predict_invalid_smiles_nan():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=5, batch_size=8)
    pred.fit(ds, verbose=False)
    out = pred.predict(["CCO", "NOT_A_MOL", "c1ccccc1"])
    assert not np.isnan(out[0])
    assert np.isnan(out[1])
    assert not np.isnan(out[2])


# ── history ───────────────────────────────────────────────────────────────────

def test_history_populated_after_fit():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=5, batch_size=8)
    pred.fit(ds, verbose=False)
    assert "train" in pred.history
    assert len(pred.history["train"]) == 5


def test_history_has_val_when_provided():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=5, batch_size=8)
    pred.fit(ds, val_dataset=ds, verbose=False)
    assert len(pred.history["val"]) == 5


# ── score ─────────────────────────────────────────────────────────────────────

def test_score_returns_dict():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=10, batch_size=8)
    pred.fit(ds, verbose=False)
    metrics = pred.score(ds)
    assert "r2" in metrics and "mae" in metrics and "rmse" in metrics


def test_score_mae_positive():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=10, batch_size=8)
    pred.fit(ds, verbose=False)
    assert pred.score(ds)["mae"] >= 0.0


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, n_layers=2, epochs=5, batch_size=8)
    pred.fit(ds, verbose=False)
    before = pred.predict(_SMILES[:3])

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.pt"
        pred.save(path)
        assert path.exists()

        pred2 = PropertyPredictor.load(path)
        after = pred2.predict(_SMILES[:3])

    np.testing.assert_allclose(before, after, rtol=1e-4)


def test_load_preserves_hparams():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=32, n_layers=2, epochs=3, batch_size=8)
    pred.fit(ds, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "m.pt"
        pred.save(path)
        pred2 = PropertyPredictor.load(path)
    assert pred2.hidden == 32
    assert pred2.n_layers == 2


def test_save_before_fit_raises():
    pred = PropertyPredictor()
    with pytest.raises(RuntimeError, match="save"):
        pred.save("/tmp/noop.pt")


# ── predict_with_uncertainty ──────────────────────────────────────────────────

def test_uncertainty_returns_mean_and_std():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, n_layers=2, epochs=5, batch_size=8, dropout=0.1)
    pred.fit(ds, verbose=False)
    mean, std = pred.predict_with_uncertainty(_SMILES[:5])
    assert mean.shape == (5,)
    assert std.shape  == (5,)


def test_uncertainty_std_non_negative():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=5, batch_size=8, dropout=0.1)
    pred.fit(ds, verbose=False)
    _, std = pred.predict_with_uncertainty(_SMILES[:5])
    assert (std[~np.isnan(std)] >= 0).all()


def test_uncertainty_invalid_smiles_nan():
    ds = _mini_dataset()
    pred = PropertyPredictor(hidden=16, epochs=5, batch_size=8, dropout=0.1)
    pred.fit(ds, verbose=False)
    mean, std = pred.predict_with_uncertainty(["CCO", "NOT_A_MOL"])
    assert np.isnan(mean[1])
    assert np.isnan(std[1])


def test_uncertainty_before_fit_raises():
    pred = PropertyPredictor()
    with pytest.raises(RuntimeError, match="fit"):
        pred.predict_with_uncertainty(["CCO"])
