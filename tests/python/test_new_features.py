"""
Tests for all 5 new feature areas:
  1. 9-feature node vectors (degree, in_ring, hybridization, chirality, mass_norm)
  2. SMARTS substructure search
  3. Murcko scaffold decomposition + scaffold_split
  4. Conformer generation + 3D descriptors
  5. CLI argument parser (no network, no file I/O needed for parser tests)
"""
import math
import numpy as np
import pytest
import torch

from molcore.molecule import Mol
from molcore import rdkit_bridge


# ── 1. Node feature dimensions ───────────────────────────────────────────────

def test_node_feature_dim_is_nine():
    mol  = Mol.from_smiles("CCO")
    data = mol.to_pyg()
    assert data.x.shape[1] == 9, f"expected 9 node features, got {data.x.shape[1]}"


def test_degree_feature():
    # propanol CCO: degrees = [1, 2, 1]
    mol  = Mol.from_smiles("CCO")
    data = mol.to_pyg()
    degrees = data.x[:, 4].tolist()
    assert degrees == [1.0, 2.0, 1.0], f"wrong degrees: {degrees}"


def test_in_ring_benzene():
    mol  = Mol.from_smiles("c1ccccc1")
    data = mol.to_pyg()
    assert all(data.x[:, 5] == 1.0), "all benzene atoms must be in_ring"


def test_in_ring_chain():
    mol  = Mol.from_smiles("CCCC")
    data = mol.to_pyg()
    assert all(data.x[:, 5] == 0.0), "chain atoms must not be in_ring"


def test_hybridization_sp3_alkane():
    mol  = Mol.from_smiles("CC")
    data = mol.to_pyg()
    assert all(data.x[:, 6] == 3.0), "alkane C must be sp3"


def test_hybridization_sp2_alkene():
    mol  = Mol.from_smiles("C=C")
    data = mol.to_pyg()
    assert all(data.x[:, 6] == 2.0), "alkene C must be sp2"


def test_hybridization_sp_alkyne():
    mol  = Mol.from_smiles("C#C")
    data = mol.to_pyg()
    assert all(data.x[:, 6] == 1.0), "alkyne C must be sp"


def test_chirality_at():
    mol  = Mol.from_smiles("N[C@H](C)C(=O)O")  # L-alanine
    data = mol.to_pyg()
    # atom 1 is [C@H]: chirality = 1
    assert data.x[1, 7].item() == 1.0, "[C@H] chirality must be 1"


def test_chirality_atat():
    mol  = Mol.from_smiles("N[C@@H](C)C(=O)O")
    data = mol.to_pyg()
    assert data.x[1, 7].item() == 2.0, "[C@@H] chirality must be 2"


def test_mass_norm_carbon():
    mol  = Mol.from_smiles("C")
    data = mol.to_pyg()
    expected = 12.011 / 100.0
    assert abs(data.x[0, 8].item() - expected) < 1e-4


# ── 2. SMARTS substructure search ────────────────────────────────────────────

def test_substructure_match_positive():
    assert rdkit_bridge.substructure_match("CC(=O)O", "C(=O)O"), "acetic acid should match carboxylic acid"


def test_substructure_match_negative():
    assert not rdkit_bridge.substructure_match("CCO", "C(=O)O"), "ethanol should not match carboxylic acid"


def test_substructure_matches_returns_tuples():
    hits = rdkit_bridge.substructure_matches("c1ccccc1CC", "c1ccccc1")
    assert len(hits) >= 1, "toluene should have one benzene ring match"
    assert all(len(t) == 6 for t in hits), "each match should be 6 atoms"


def test_filter_by_smarts_keep():
    acids = rdkit_bridge.filter_by_smarts(
        ["CCO", "CC(=O)O", "c1ccccc1C(=O)O", "CCCC"],
        "C(=O)O",
    )
    assert set(acids) == {"CC(=O)O", "c1ccccc1C(=O)O"}


def test_filter_by_smarts_invert():
    non_acids = rdkit_bridge.filter_by_smarts(
        ["CCO", "CC(=O)O", "CCCC"],
        "C(=O)O",
        invert=True,
    )
    assert set(non_acids) == {"CCO", "CCCC"}


def test_invalid_smarts_raises():
    with pytest.raises(ValueError, match="Invalid SMARTS"):
        rdkit_bridge.substructure_match("CCO", "not_valid_smarts$$$$")


def test_mol_matches_method():
    mol = Mol.from_smiles("c1ccccc1")
    assert mol.matches("c1ccccc1"), "benzene matches benzene SMARTS"
    assert not mol.matches("[NH2]"), "benzene does not match amine"


def test_mol_find_substructures():
    mol = Mol.from_smiles("c1ccc(cc1)c2ccccc2")  # biphenyl
    hits = mol.find_substructures("c1ccccc1")
    assert len(hits) >= 2, "biphenyl should have ≥2 benzene ring matches"


# ── 3. Murcko scaffold + scaffold_split ──────────────────────────────────────

def test_murcko_scaffold_benzene():
    sc = rdkit_bridge.murcko_scaffold("c1ccccc1")
    assert sc == "c1ccccc1", f"benzene scaffold is benzene, got {sc!r}"


def test_murcko_scaffold_toluene():
    sc = rdkit_bridge.murcko_scaffold("Cc1ccccc1")
    assert sc == "c1ccccc1", f"toluene scaffold is benzene, got {sc!r}"


def test_murcko_scaffold_generic():
    sc = rdkit_bridge.murcko_scaffold("Cc1ccccc1", generic=True)
    # Generic scaffold replaces all atoms with C — 6-membered ring
    assert "C" in sc and "c" not in sc, f"generic scaffold should be all-carbon: {sc!r}"


def test_murcko_scaffold_no_ring():
    sc = rdkit_bridge.murcko_scaffold("CCCC")
    assert sc == "", f"acyclic SMILES should return empty scaffold, got {sc!r}"


def test_mol_scaffold_method():
    aspirin = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    scaffold = aspirin.scaffold()
    assert isinstance(scaffold, Mol)
    assert "c" in scaffold.smiles, "aspirin scaffold should contain aromatic ring"


def test_scaffold_split_sizes():
    smiles = [
        "Cc1ccccc1", "Clc1ccccc1", "Fc1ccccc1",   # benzene scaffolds
        "c1ccncc1", "Cc1ccncc1",                    # pyridine scaffolds
        "CCCO", "CCCCO", "CCCCCO",                  # aliphatic
    ]
    train, val, test = rdkit_bridge.scaffold_split(smiles, train_frac=0.6, val_frac=0.2)
    assert len(train) + len(val) + len(test) == len(smiles)
    assert len(train) > 0 and len(val) >= 0 and len(test) >= 0


# ── 4. Conformer generation + 3D descriptors ─────────────────────────────────

def test_conformer_shape():
    mol  = Mol.from_smiles("CCO")
    confs = mol.conformers(n_confs=1)
    assert len(confs) == 1, "requested 1 conformer"
    assert confs[0].shape == (3, 3), "ethanol: 3 heavy atoms × 3 coords"


def test_conformer_multiple():
    mol   = Mol.from_smiles("c1ccccc1")
    confs = mol.conformers(n_confs=3)
    assert len(confs) == 3


def test_conformer_dtype():
    mol  = Mol.from_smiles("CCO")
    conf = mol.conformers()[0]
    assert conf.dtype == np.float64


def test_generate_conformers_rdkit_bridge():
    confs = rdkit_bridge.generate_conformers("CC(=O)O", n_confs=2)
    assert len(confs) == 2
    assert confs[0].shape[1] == 3


def test_3d_descriptors_keys():
    desc = rdkit_bridge.calc_descriptors_3d("c1ccccc1")
    required = {"pmi1", "pmi2", "pmi3", "asphericity", "eccentricity",
                "npr1", "npr2", "radius_of_gyration"}
    assert required.issubset(desc.keys())


def test_3d_descriptors_benzene_values():
    desc = rdkit_bridge.calc_descriptors_3d("c1ccccc1")
    # benzene is flat → asphericity ≈ 0.25 (oblate disc)
    assert desc["asphericity"] > 0.0
    # PMI1 < PMI2 < PMI3 always holds
    assert desc["pmi1"] <= desc["pmi2"] <= desc["pmi3"]


def test_mol_descriptors_3d():
    mol  = Mol.from_smiles("CCO")
    desc = mol.descriptors_3d()
    assert isinstance(desc, dict)
    assert "asphericity" in desc


# ── 5. CLI parser (no subprocess, no file I/O) ───────────────────────────────

def test_cli_featurize_help():
    from molcore.cli import build_parser
    parser = build_parser()
    # Parsing --help would sys.exit; just verify the subcommand exists
    sub_names = [a.option_string for a in parser._subparsers._actions
                 if hasattr(a, "option_string")] if False else []
    # Simpler: parse the featurize subcommand with required args
    args = parser.parse_args(["featurize", "dummy.smi"])
    assert args.command == "featurize"
    assert args.backend == "rust"
    assert args.nbits == 2048


def test_cli_screen_defaults():
    from molcore.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["screen", "q.smi", "lib.smi"])
    assert args.top_k == 100
    assert args.backend == "rust"


def test_cli_benchmark_defaults():
    from molcore.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["benchmark"])
    assert args.n == 1000
    assert args.repeats == 3


def test_cli_scaffold_split_defaults():
    from molcore.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["scaffold-split", "mols.smi"])
    assert args.train_frac == 0.8
    assert args.val_frac == 0.1
    assert args.seed == 42
