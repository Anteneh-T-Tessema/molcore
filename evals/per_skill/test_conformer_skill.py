"""
Eval: conformer skill — 3D coordinate generation and shape descriptor correctness.
"""
import pytest
import numpy as np
from tools.local.conformer import run


# ── descriptors_3d mode ───────────────────────────────────────────────────────

def test_descriptors_3d_keys_present():
    result = run("c1ccccc1", mode="descriptors_3d")
    assert "error" not in result
    required = {"pmi1", "pmi2", "pmi3", "asphericity", "eccentricity",
                "npr1", "npr2", "radius_of_gyration"}
    assert required.issubset(result["descriptors"].keys())


def test_descriptors_3d_pmi_ordered():
    result = run("c1ccccc1", mode="descriptors_3d")
    d = result["descriptors"]
    assert d["pmi1"] <= d["pmi2"] <= d["pmi3"], "PMI must be non-decreasing"


def test_descriptors_3d_asphericity_positive():
    result = run("c1ccccc1", mode="descriptors_3d")
    assert result["descriptors"]["asphericity"] > 0.0


def test_descriptors_3d_benzene_shape():
    result = run("c1ccccc1", mode="descriptors_3d")
    d = result["descriptors"]
    # Benzene is a flat disc: asphericity ≈ 0.25, NPR1 ≈ 0.46
    assert 0.1 < d["asphericity"] < 0.5, f"benzene asphericity out of range: {d['asphericity']:.3f}"
    assert 0.3 < d["npr1"] < 0.7, f"benzene NPR1 out of range: {d['npr1']:.3f}"


def test_descriptors_3d_accepts_list():
    result = run(["c1ccccc1"], mode="descriptors_3d")
    assert "error" not in result
    assert "descriptors" in result


def test_descriptors_3d_smiles_echoed():
    result = run("CCO", mode="descriptors_3d")
    assert result["smiles"] == "CCO"


# ── coordinates mode ─────────────────────────────────────────────────────────

def test_coordinates_shape_ethanol():
    result = run("CCO", mode="coordinates", n_confs=1)
    assert "error" not in result
    assert result["n_conformers"] == 1
    assert result["n_atoms"] == 3     # 3 heavy atoms
    coords = np.array(result["coordinates"][0])
    assert coords.shape == (3, 3)


def test_coordinates_multiple_confs():
    result = run("c1ccccc1", mode="coordinates", n_confs=3)
    assert result["n_conformers"] == 3
    assert len(result["coordinates"]) == 3


def test_coordinates_3d_not_flat():
    # Aspirin is non-planar after optimization — at least one z-coord should differ
    result = run("CC(=O)Oc1ccccc1C(=O)O", mode="coordinates", n_confs=1)
    coords = np.array(result["coordinates"][0])
    z_range = coords[:, 2].max() - coords[:, 2].min()
    assert z_range > 0.0, "3D coords should not all be in one plane"


# ── batch_descriptors mode ────────────────────────────────────────────────────

def test_batch_descriptors_multiple():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    result = run(smiles, mode="batch_descriptors")
    assert result["n_molecules"] == 3
    assert len(result["results"]) == 3
    for r in result["results"]:
        assert "error" not in r or "descriptors" in r  # at least one should succeed


def test_batch_descriptors_keys_per_mol():
    result = run(["CCO", "c1ccccc1"], mode="batch_descriptors")
    for r in result["results"]:
        if "error" not in r:
            assert "pmi1" in r["descriptors"]


def test_batch_descriptors_single_string():
    result = run("CCO", mode="batch_descriptors")
    assert result["n_molecules"] == 1


# ── invalid mode ──────────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        run("CCO", mode="bad_mode")
