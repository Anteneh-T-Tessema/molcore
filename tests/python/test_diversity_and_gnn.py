"""
Tests for:
  - diversity_pick (rdkit_bridge + MolDataset.diversity_pick)
  - PropertyPredictor model_type: gcn, gat, gin
  - checkpoint backward-compatibility for model_type
"""
import numpy as np
import pytest

SMILES = [
    "CCO",                  # ethanol
    "CCCO",                 # propanol
    "c1ccccc1",             # benzene
    "c1ccccc1O",            # phenol
    "CC(=O)O",              # acetic acid
    "c1ccc(F)cc1",          # fluorobenzene
    "c1ccc(Cl)cc1",         # chlorobenzene
    "CC(C)O",               # isopropanol
    "c1ccc(N)cc1",          # aniline
    "c1ccc(Br)cc1",         # bromobenzene
]


# ---------------------------------------------------------------------------
# diversity_pick (function)
# ---------------------------------------------------------------------------

def test_diversity_pick_returns_n_indices():
    from molcore.rdkit_bridge import diversity_pick
    indices = diversity_pick(SMILES, n=4)
    assert len(indices) == 4


def test_diversity_pick_indices_in_range():
    from molcore.rdkit_bridge import diversity_pick
    indices = diversity_pick(SMILES, n=5)
    assert all(0 <= i < len(SMILES) for i in indices)


def test_diversity_pick_no_duplicates():
    from molcore.rdkit_bridge import diversity_pick
    indices = diversity_pick(SMILES, n=6)
    assert len(set(indices)) == len(indices)


def test_diversity_pick_n_larger_than_valid_clamps():
    from molcore.rdkit_bridge import diversity_pick
    indices = diversity_pick(SMILES, n=100)
    assert len(indices) == len(SMILES)


def test_diversity_pick_invalid_smiles_excluded():
    from molcore.rdkit_bridge import diversity_pick
    smiles_with_bad = SMILES[:3] + ["NOT_VALID"] + SMILES[3:]
    indices = diversity_pick(smiles_with_bad, n=4)
    assert all(smiles_with_bad[i] != "NOT_VALID" for i in indices)


def test_diversity_pick_empty():
    from molcore.rdkit_bridge import diversity_pick
    assert diversity_pick([], n=5) == []


def test_diversity_pick_picks_structurally_different_molecules():
    from molcore.rdkit_bridge import diversity_pick
    # Aromatics vs aliphatics — a diverse pick of 2 should not be two similar aliphatics
    indices = diversity_pick(SMILES, n=2, seed=0)
    picked = [SMILES[i] for i in indices]
    # One should be aromatic, one aliphatic — they differ maximally in Tanimoto space
    aromatic = any("c" in s for s in picked)
    aliphatic = any("c" not in s for s in picked)
    assert aromatic and aliphatic, f"Expected diverse pick, got: {picked}"


# ---------------------------------------------------------------------------
# MolDataset.diversity_pick
# ---------------------------------------------------------------------------

def test_dataset_diversity_pick_returns_dataset():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    picked = ds.diversity_pick(n=5)
    assert len(picked) == 5


def test_dataset_diversity_pick_preserves_labels():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.labels = np.arange(len(SMILES), dtype=np.float32)
    picked = ds.diversity_pick(n=4)
    assert picked.labels is not None
    assert len(picked.labels) == 4


def test_dataset_diversity_pick_does_not_mutate_original():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    _ = ds.diversity_pick(n=3)
    assert len(ds) == len(SMILES)


# ---------------------------------------------------------------------------
# PropertyPredictor: model_type gcn / gat / gin
# ---------------------------------------------------------------------------

TRAIN_SMILES = [
    "CCO", "CCCO", "c1ccccc1", "c1ccccc1O", "CC(=O)O",
    "c1ccc(F)cc1", "CC(C)O", "CCC(=O)O", "Cc1ccccc1", "c1cccnc1",
]
TRAIN_LABELS = np.array(
    [-0.14, 0.25, 1.68, 1.46, -0.17, 1.15, 0.05, 0.33, 2.11, -0.91],
    dtype=np.float32,
)


def _make_ds(smiles=None, labels=None):
    from molcore.io import MolDataset
    smiles = smiles or TRAIN_SMILES
    labels = labels if labels is not None else TRAIN_LABELS
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    ds.labels = labels
    return ds


@pytest.mark.parametrize("model_type", ["gcn", "gat", "gin"])
def test_model_type_trains_and_predicts(model_type):
    from molcore.predictor import PropertyPredictor
    ds = _make_ds()
    pred = PropertyPredictor(hidden=32, n_layers=2, epochs=5, model_type=model_type)
    pred.fit(ds, verbose=False)
    preds = pred.predict(TRAIN_SMILES[:3])
    assert preds.shape == (3,)
    assert not np.any(np.isnan(preds))


@pytest.mark.parametrize("model_type", ["gcn", "gat", "gin"])
def test_model_type_save_load_roundtrip(tmp_path, model_type):
    from molcore.predictor import PropertyPredictor
    ds = _make_ds()
    pred = PropertyPredictor(hidden=32, n_layers=2, epochs=5, model_type=model_type)
    pred.fit(ds, verbose=False)

    p = tmp_path / f"model_{model_type}.pt"
    pred.save(str(p))

    loaded = PropertyPredictor.load(str(p))
    assert loaded.model_type == model_type
    preds = loaded.predict(TRAIN_SMILES[:3])
    assert not np.any(np.isnan(preds))


def test_invalid_model_type_raises():
    from molcore.predictor import PropertyPredictor
    from molcore.io import MolDataset
    ds = _make_ds()
    pred = PropertyPredictor(model_type="transformer", epochs=1)
    with pytest.raises(ValueError, match="model_type"):
        pred.fit(ds, verbose=False)


def test_load_old_checkpoint_defaults_to_gcn(tmp_path):
    """Checkpoints without model_type (pre-v0.2) load as GCN — backward compat."""
    import torch
    from molcore.predictor import PropertyPredictor, _MolGNN
    ds = _make_ds()
    pred = PropertyPredictor(hidden=32, n_layers=2, epochs=5, model_type="gcn")
    pred.fit(ds, verbose=False)

    # Manually save without model_type in hparams to simulate old checkpoint
    p = tmp_path / "old_checkpoint.pt"
    torch.save({
        "state_dict": pred._model.state_dict(),
        "hparams": {"hidden": 32, "n_layers": 2, "dropout": 0.1, "n_outputs": 1},
        "history": {},
    }, str(p))

    loaded = PropertyPredictor.load(str(p))
    assert loaded.model_type == "gcn"
    preds = loaded.predict(TRAIN_SMILES[:2])
    assert not np.any(np.isnan(preds))
