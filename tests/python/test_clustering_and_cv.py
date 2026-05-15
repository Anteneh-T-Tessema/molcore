"""
Tests for:
  - butina_cluster (rdkit_bridge + MolDataset.cluster)
  - MolDataset.kfold
  - MolDataset.scaffold_kfold
  - PropertyPredictor.tune (Optuna; skipped if optuna not installed)
"""
import numpy as np
import pytest

SMILES = [
    "CCO",                           # ethanol
    "CCCO",                          # propanol
    "CCCCO",                         # butanol  (similar to propanol/ethanol)
    "c1ccccc1",                      # benzene
    "c1ccccc1O",                     # phenol   (similar to benzene)
    "c1ccc(F)cc1",                   # fluorobenzene
    "CC(=O)O",                       # acetic acid
    "CCC(=O)O",                      # propionic acid
]


# ---------------------------------------------------------------------------
# butina_cluster (function)
# ---------------------------------------------------------------------------

def test_butina_returns_one_id_per_molecule():
    from molcore.rdkit_bridge import butina_cluster
    ids = butina_cluster(SMILES)
    assert len(ids) == len(SMILES)


def test_butina_all_ids_non_negative():
    from molcore.rdkit_bridge import butina_cluster
    ids = butina_cluster(SMILES)
    assert all(i >= 0 for i in ids), f"Expected no -1 for valid SMILES, got {ids}"


def test_butina_invalid_smiles_gets_minus_one():
    from molcore.rdkit_bridge import butina_cluster
    ids = butina_cluster(["CCO", "NOT_VALID", "c1ccccc1"])
    assert ids[1] == -1


def test_butina_tight_cutoff_more_clusters():
    from molcore.rdkit_bridge import butina_cluster
    ids_loose = butina_cluster(SMILES, cutoff=0.8)
    ids_tight = butina_cluster(SMILES, cutoff=0.2)
    assert max(ids_tight) >= max(ids_loose)


def test_butina_similar_molecules_same_cluster():
    from molcore.rdkit_bridge import butina_cluster
    # Very loose cutoff: all aliphatic alcohols should cluster together
    alcs = ["CCO", "CCCO", "CCCCO", "CCCCCO"]
    ids = butina_cluster(alcs, cutoff=0.8)
    assert len(set(ids)) == 1, f"Expected 1 cluster with loose cutoff, got {set(ids)}"


def test_butina_empty_input():
    from molcore.rdkit_bridge import butina_cluster
    assert butina_cluster([]) == []


# ---------------------------------------------------------------------------
# MolDataset.cluster
# ---------------------------------------------------------------------------

def test_dataset_cluster_adds_metadata():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    clustered = ds.cluster(cutoff=0.6)
    assert "cluster_id" in clustered.metadata
    assert len(clustered.metadata["cluster_id"]) == len(SMILES)


def test_dataset_cluster_does_not_mutate_original():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    _ = ds.cluster()
    assert "cluster_id" not in ds.metadata


# ---------------------------------------------------------------------------
# MolDataset.kfold
# ---------------------------------------------------------------------------

def test_kfold_returns_k_pairs():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    folds = ds.kfold(k=4)
    assert len(folds) == 4


def test_kfold_val_sizes_partition_dataset():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    folds = ds.kfold(k=4, seed=0)
    total_val = sum(len(val) for _, val in folds)
    assert total_val == len(SMILES)


def test_kfold_train_plus_val_equals_n():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    for train, val in ds.kfold(k=4):
        assert len(train) + len(val) == len(SMILES)


def test_kfold_val_folds_are_disjoint():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    folds = ds.kfold(k=4, seed=7)
    val_smiles_sets = [set(val.smiles) for _, val in folds]
    for i in range(len(val_smiles_sets)):
        for j in range(i + 1, len(val_smiles_sets)):
            overlap = val_smiles_sets[i] & val_smiles_sets[j]
            assert not overlap, f"Val folds {i} and {j} share SMILES: {overlap}"


def test_kfold_k_too_small_raises():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    with pytest.raises(ValueError, match="k must be"):
        ds.kfold(k=1)


def test_kfold_k_larger_than_n_raises():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(["CCO", "c1ccccc1"], compute_fps=False, compute_desc=False)
    with pytest.raises(ValueError):
        ds.kfold(k=5)


def test_kfold_preserves_labels():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.labels = np.arange(len(SMILES), dtype=np.float32)
    for train, val in ds.kfold(k=4):
        assert train.labels is not None
        assert val.labels is not None


# ---------------------------------------------------------------------------
# MolDataset.scaffold_kfold
# ---------------------------------------------------------------------------

def test_scaffold_kfold_returns_k_pairs():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    folds = ds.scaffold_kfold(k=3)
    assert len(folds) == 3


def test_scaffold_kfold_all_molecules_appear_in_val():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    folds = ds.scaffold_kfold(k=4, seed=0)
    all_val = []
    for _, val in folds:
        all_val.extend(val.smiles)
    assert sorted(all_val) == sorted(SMILES)


def test_scaffold_kfold_train_plus_val_equals_n():
    from molcore.io import MolDataset
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    for train, val in ds.scaffold_kfold(k=3):
        assert len(train) + len(val) == len(SMILES)


# ---------------------------------------------------------------------------
# PropertyPredictor.tune  (skipped without optuna)
# ---------------------------------------------------------------------------

def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_optuna = pytest.mark.skipif(not _has_optuna(), reason="optuna not installed")


@skip_no_optuna
def test_tune_improves_or_matches_default():
    from molcore.io import MolDataset
    from molcore.predictor import PropertyPredictor
    smiles = [
        "CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1O",
        "CC(=O)O", "c1ccc(F)cc1", "CC(C)O", "CCC(=O)O", "Cc1ccccc1",
    ]
    logp = np.array([-0.14, 0.25, 0.88, 1.68, 1.46, -0.17, 1.15, 0.05, 0.33, 2.11],
                    dtype=np.float32)
    train = MolDataset.from_smiles(smiles[:7], compute_fps=False, compute_desc=False)
    train.labels = logp[:7]
    val = MolDataset.from_smiles(smiles[7:], compute_fps=False, compute_desc=False)
    val.labels = logp[7:]

    pred = PropertyPredictor(epochs=30)
    pred.tune(train, val, n_trials=5, verbose=False)

    preds = pred.predict(smiles)
    assert not np.any(np.isnan(preds)), "tune() produced NaN predictions"
    assert pred.hidden in [32, 64, 128, 256]
    assert pred.n_layers in [2, 3, 4]


@skip_no_optuna
def test_tune_returns_self():
    from molcore.io import MolDataset
    from molcore.predictor import PropertyPredictor
    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "c1ccccc1O", "CC(C)O"]
    logp = np.array([-0.14, 0.25, 1.68, -0.17, 1.46, 0.05], dtype=np.float32)
    train = MolDataset.from_smiles(smiles[:4], compute_fps=False, compute_desc=False)
    train.labels = logp[:4]
    val = MolDataset.from_smiles(smiles[4:], compute_fps=False, compute_desc=False)
    val.labels = logp[4:]
    pred = PropertyPredictor(epochs=20)
    result = pred.tune(train, val, n_trials=3, verbose=False)
    assert result is pred


def test_tune_raises_without_optuna(monkeypatch):
    """tune() gives a clear ImportError when optuna is absent."""
    import sys
    from molcore.predictor import PropertyPredictor
    from molcore.io import MolDataset

    smiles = ["CCO", "c1ccccc1"]
    train = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    train.labels = np.array([1.0, 2.0], dtype=np.float32)

    pred = PropertyPredictor(epochs=5)

    # Temporarily hide optuna from the import system
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

    import builtins
    original = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("No module named 'optuna'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="optuna"):
        pred.tune(train, train, n_trials=1)
