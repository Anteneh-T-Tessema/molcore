"""
Per-skill eval: similarity_search skill — validates SKILL.md contract.
"""
import json
import pathlib
import pytest
import torch
import molcore
from molcore._molcore import tanimoto_matrix

SKILL_DIR = pathlib.Path(__file__).parents[2] / "skills" / "similarity_search"
EXAMPLES   = [json.loads(l) for l in (SKILL_DIR / "examples.jsonl").read_text().splitlines() if l.strip()]


def test_skill_md_exists():
    assert (SKILL_DIR / "SKILL.md").exists()


def test_self_similarity_is_one():
    fps = molcore.featurize_smiles(["c1ccccc1"]).numpy()
    sim = tanimoto_matrix(fps, fps)
    assert abs(sim[0, 0] - 1.0) < 1e-5


def test_different_molecules_lt_one():
    fps = molcore.featurize_smiles(["CCO", "c1ccccc1"]).numpy()
    sim = tanimoto_matrix(fps, fps)
    assert sim[0, 1] < 1.0
    assert sim[1, 0] < 1.0


def test_matrix_shape():
    q_fps = molcore.featurize_smiles(["CCO", "CC(=O)O"]).numpy()
    l_fps = molcore.featurize_smiles(["CCO", "c1ccccc1", "CC"]).numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)
    assert sim.shape == (2, 3)


def test_scores_in_range():
    fps = molcore.featurize_smiles(["CCO", "c1ccccc1", "CC(=O)O"]).numpy()
    sim = tanimoto_matrix(fps, fps)
    assert float(sim.min()) >= 0.0
    assert float(sim.max()) <= 1.0


def test_symmetry():
    fps = molcore.featurize_smiles(["CCO", "c1ccccc1", "CC(=O)O"]).numpy()
    sim = tanimoto_matrix(fps, fps)
    import numpy as np
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)
