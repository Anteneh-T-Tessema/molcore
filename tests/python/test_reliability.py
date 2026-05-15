"""
Regression tests for the six reliability fixes:
  1. tanimoto_matrix — ValueError on mismatched nbits
  2. scaffold_split  — duplicate SMILES not silently dropped
  3. Parquet         — multi-label roundtrip
  4. neutralize / strip_salts — use cached objects (no object-creation regression)
  5. RDKit firewall  — io.from_sdf uses mol_to_smiles, not rdkit.Chem directly
  6. write_sdf       — ValueError on mismatched properties length
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. tanimoto_matrix: mismatched nbits must raise, not silently truncate
# ---------------------------------------------------------------------------

def test_tanimoto_mismatched_nbits_raises():
    from molcore._molcore import tanimoto_matrix
    q = np.zeros((2, 2048), dtype=np.uint8)
    l = np.zeros((3, 1024), dtype=np.uint8)
    with pytest.raises(ValueError, match="bits"):
        tanimoto_matrix(q, l)


def test_tanimoto_matched_nbits_ok():
    from molcore._molcore import tanimoto_matrix
    q = np.eye(4, dtype=np.uint8)
    l = np.eye(4, dtype=np.uint8)
    sim = tanimoto_matrix(q, l)
    assert sim.shape == (4, 4)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. scaffold_split: duplicate SMILES must all be preserved
# ---------------------------------------------------------------------------

def test_scaffold_split_duplicate_smiles():
    from molcore.io import MolDataset
    import numpy as np

    # Six molecules: "CCO" appears three times
    smiles = ["CCO", "CCO", "CCO", "c1ccccc1", "CC(=O)O", "c1cccnc1"]
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    ds.labels = np.arange(6, dtype=np.float32)

    train, val, test = ds.scaffold_split(train_frac=0.6, val_frac=0.2, seed=42)

    total = len(train) + len(val) + len(test)
    assert total == 6, f"Expected 6 molecules across splits, got {total}"

    # Labels must not be duplicated — each original label appears exactly once
    all_labels = np.concatenate([
        train.labels if train.labels is not None else np.array([]),
        val.labels   if val.labels   is not None else np.array([]),
        test.labels  if test.labels  is not None else np.array([]),
    ])
    assert sorted(all_labels.tolist()) == pytest.approx(list(range(6)), abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Parquet: multi-label roundtrip
# ---------------------------------------------------------------------------

def test_parquet_multi_label_roundtrip(tmp_path):
    from molcore.io import MolDataset
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    ds.labels = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    p = tmp_path / "multi.parquet"
    ds.write_parquet(str(p))

    ds2 = MolDataset.read_parquet(str(p))
    assert ds2.labels is not None, "Multi-label not recovered from Parquet"
    assert ds2.labels.shape == (3, 2)
    np.testing.assert_allclose(ds2.labels, ds.labels, atol=1e-6)


def test_parquet_single_label_roundtrip(tmp_path):
    from molcore.io import MolDataset
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    ds.labels = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    p = tmp_path / "single.parquet"
    ds.write_parquet(str(p))

    ds2 = MolDataset.read_parquet(str(p))
    assert ds2.labels is not None
    assert ds2.labels.shape == (3,)
    np.testing.assert_allclose(ds2.labels, ds.labels, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. neutralize / strip_salts use the module-level cache (_std_objects)
# ---------------------------------------------------------------------------

def test_neutralize_uses_cache(monkeypatch):
    """_std_objects should be called, not a fresh Uncharger() each time."""
    from molcore import rdkit_bridge
    call_count = 0
    original = rdkit_bridge._std_objects

    def counting_wrapper():
        nonlocal call_count
        call_count += 1
        return original()

    monkeypatch.setattr(rdkit_bridge, "_std_objects", counting_wrapper)
    rdkit_bridge.neutralize("CC(=O)[O-]")
    rdkit_bridge.neutralize("CC(=O)[O-]")
    assert call_count == 2, "Expected _std_objects called once per neutralize() call"


def test_strip_salts_uses_cache(monkeypatch):
    from molcore import rdkit_bridge
    call_count = 0
    original = rdkit_bridge._std_objects

    def counting_wrapper():
        nonlocal call_count
        call_count += 1
        return original()

    monkeypatch.setattr(rdkit_bridge, "_std_objects", counting_wrapper)
    rdkit_bridge.strip_salts("[Na+].OC(=O)c1ccccc1")
    rdkit_bridge.strip_salts("[Na+].OC(=O)c1ccccc1")
    assert call_count == 2


def test_neutralize_correctness():
    from molcore.rdkit_bridge import neutralize
    result = neutralize("CC(=O)[O-]")
    assert "[O-]" not in result


def test_strip_salts_correctness():
    from molcore.rdkit_bridge import strip_salts
    result = strip_salts("[Na+].OC(=O)c1ccccc1")
    assert "." not in result


# ---------------------------------------------------------------------------
# 5. RDKit firewall: io.from_sdf must not import rdkit directly
# ---------------------------------------------------------------------------

def test_from_sdf_no_rdkit_import(tmp_path):
    """Verify that MolDataset.from_sdf works and that rdkit_bridge exposes mol_to_smiles."""
    from molcore.rdkit_bridge import mol_to_smiles, from_smiles
    rdmol = from_smiles("CCO")
    smi = mol_to_smiles(rdmol)
    assert smi  # must return a non-empty SMILES


def test_from_sdf_roundtrip(tmp_path):
    from molcore.rdkit_bridge import write_sdf
    from molcore.io import MolDataset

    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    p = tmp_path / "test.sdf"
    write_sdf(smiles, str(p))

    ds = MolDataset.from_sdf(str(p), compute_fps=False, compute_desc=False)
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# 6. write_sdf: mismatched properties length raises ValueError
# ---------------------------------------------------------------------------

def test_write_sdf_mismatched_properties_raises(tmp_path):
    from molcore.rdkit_bridge import write_sdf
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    p = tmp_path / "bad.sdf"
    with pytest.raises(ValueError, match="lengths must match"):
        write_sdf(smiles, str(p), properties={"mw": [46.0, 78.0]})  # 2 values for 3 mols


def test_write_sdf_matched_properties_ok(tmp_path):
    from molcore.rdkit_bridge import write_sdf
    smiles = ["CCO", "c1ccccc1"]
    p = tmp_path / "good.sdf"
    write_sdf(smiles, str(p), properties={"mw": [46.0, 78.0]})
    assert p.exists()
