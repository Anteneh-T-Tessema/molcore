"""
Eval: scaffold skill — Murcko decomposition, split, and clustering correctness.
"""
import pytest
from tools.local.scaffold import run

BENZENE_SERIES = ["Cc1ccccc1", "Clc1ccccc1", "Fc1ccccc1", "Brc1ccccc1"]
PYRIDINE_SERIES = ["c1ccncc1", "Cc1ccncc1"]
ACYCLICS = ["CCCO", "CCCCO", "CCCCCO"]
MIXED = BENZENE_SERIES + PYRIDINE_SERIES + ACYCLICS


# ── scaffold mode ─────────────────────────────────────────────────────────────

def test_scaffold_toluene_is_benzene():
    result = run(["Cc1ccccc1"], mode="scaffold")
    assert result["scaffolds"] == ["c1ccccc1"]


def test_scaffold_acyclic_empty():
    result = run(["CCCC"], mode="scaffold")
    assert result["scaffolds"] == [""]


def test_scaffold_same_scaffold_for_series():
    result = run(BENZENE_SERIES, mode="scaffold")
    scaffolds = set(result["scaffolds"])
    assert len(scaffolds) == 1, f"all benzene-substituted should share scaffold, got {scaffolds}"


def test_scaffold_generic_all_carbon():
    result = run(["Cc1ccncc1"], mode="scaffold", generic=True)
    sc = result["scaffolds"][0]
    assert "n" not in sc and "N" not in sc, f"generic scaffold should have no heteroatoms: {sc!r}"


def test_scaffold_count():
    result = run(MIXED, mode="scaffold")
    assert result["n_molecules"] == len(MIXED)
    assert len(result["scaffolds"]) == len(MIXED)


# ── split mode ────────────────────────────────────────────────────────────────

def test_split_total_preserved():
    result = run(MIXED, mode="split", train_frac=0.7, val_frac=0.15)
    total = result["n_train"] + result["n_val"] + result["n_test"]
    assert total == len(MIXED), f"split lost molecules: {total} vs {len(MIXED)}"


def test_split_no_duplicates():
    result = run(MIXED, mode="split")
    all_split = result["train"] + result["val"] + result["test"]
    assert len(all_split) == len(set(all_split)), "duplicate molecules across splits"


def test_split_train_larger_than_val():
    result = run(MIXED, mode="split", train_frac=0.7, val_frac=0.15)
    assert result["n_train"] >= result["n_val"]


def test_split_fractions_respected_approximately():
    smiles = MIXED * 5   # 55 molecules
    result = run(smiles, mode="split", train_frac=0.8, val_frac=0.1)
    train_frac = result["n_train"] / len(smiles)
    assert 0.5 <= train_frac <= 1.0, f"train fraction {train_frac:.2f} wildly off"


# ── cluster mode ──────────────────────────────────────────────────────────────

def test_cluster_groups_same_scaffold():
    result = run(BENZENE_SERIES + PYRIDINE_SERIES, mode="cluster")
    clusters = result["clusters"]
    # benzene and pyridine scaffolds should be in separate clusters
    assert result["n_scaffolds"] >= 2


def test_cluster_acyclics_grouped_separately():
    result = run(ACYCLICS + BENZENE_SERIES, mode="cluster")
    # acyclics get "__acyclic__" cluster key
    assert "__acyclic__" in result["clusters"]
    assert len(result["clusters"]["__acyclic__"]) == len(ACYCLICS)


def test_cluster_total_molecules():
    result = run(MIXED, mode="cluster")
    total = sum(len(v) for v in result["clusters"].values())
    assert total == len(MIXED)


# ── invalid mode ──────────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        run(["CCO"], mode="bad_mode")
