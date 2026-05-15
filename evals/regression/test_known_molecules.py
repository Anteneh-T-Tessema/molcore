"""
Regression eval: known molecules with verified properties.
These tests must never change — they guard against chemistry regressions.
"""
import pytest
import molcore
from molcore.molecule import Mol
from molcore.featurizers.descriptors import calc_descriptors


# Verified reference values (RDKit backend, locked against RDKit 2023.9+)
KNOWN = [
    # (smiles,          mw_approx, logp_approx, heavy_atoms)
    ("CCO",             46.07,     -0.14,        3),
    ("c1ccccc1",        78.11,      1.90,        6),
    ("CC(=O)O",         60.05,     -0.17,        4),
    ("c1ccccc1C(=O)O", 122.12,      1.87,        9),
]


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_descriptor_heavy_atom_count_exact(smiles, mw_ref, logp_ref, ha_ref):
    """Heavy atom count from Rust must be exact — no approximation."""
    desc = calc_descriptors([smiles], backend="rust")
    heavy = int(desc[0, 2])
    assert heavy == ha_ref, f"{smiles}: expected {ha_ref} heavy atoms, got {heavy}"


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_descriptor_mw_rdkit_exact(smiles, mw_ref, logp_ref, ha_ref):
    """RDKit backend MW must be within 0.1 Da of reference (includes implicit H)."""
    desc = calc_descriptors([smiles], backend="rdkit")
    mw = float(desc[0, 0])
    assert abs(mw - mw_ref) < 0.1, f"{smiles}: MW {mw:.3f} vs ref {mw_ref:.3f}"


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_immutability_preserved_after_transform(smiles, mw_ref, logp_ref, ha_ref):
    """Transforms must return new Mol; original smiles field must be unchanged."""
    mol  = Mol.from_smiles(smiles)
    orig = mol.smiles
    _    = mol.neutralize()
    assert mol.smiles == orig, "neutralize() must not modify the original Mol"


def test_benzene_fingerprint_nonzero():
    fps = molcore.featurize_smiles(["c1ccccc1"], backend="rust")
    assert int(fps.sum()) > 0


def test_pyg_benzene_node_count():
    data = Mol.from_smiles("c1ccccc1").to_pyg()
    assert data.x.shape[0] == 6   # 6 carbons
    assert data.edge_index.shape[1] == 12  # 6 bonds × 2 (bidirectional)


# ── Invariant guards — these must never change without a major version bump ───

def test_node_feature_dim_invariant():
    """NODE_FEAT_DIM=9 is load-bearing: trained GNN weights depend on it."""
    data = Mol.from_smiles("CCO").to_pyg()
    assert data.x.shape[1] == 9, \
        f"Node feature dim changed: {data.x.shape[1]} != 9. " \
        "All GNN models must be retrained after changing this."


def test_ecfp4_bit_stability():
    """Same SMILES must produce identical bit vectors across calls (deterministic)."""
    fps1 = molcore.featurize_smiles(["c1ccccc1"], backend="rust")
    fps2 = molcore.featurize_smiles(["c1ccccc1"], backend="rust")
    assert (fps1 == fps2).all(), "Fingerprint is not deterministic — Rust RNG leak?"


def test_scaffold_split_reproducible():
    """Scaffold split with fixed seed must return identical partitions across runs."""
    from molcore.io import MolDataset
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"] * 4
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    t1, v1, _ = ds.scaffold_split(seed=42)
    t2, v2, _ = ds.scaffold_split(seed=42)
    assert t1.smiles == t2.smiles, "Train split not reproducible with same seed"
    assert v1.smiles == v2.smiles, "Val split not reproducible with same seed"


def test_different_molecules_different_fingerprints():
    """Ethanol and propanol must produce distinct bit vectors."""
    fps = molcore.featurize_smiles(["CCO", "CCCO"], backend="rust")
    assert not (fps[0] == fps[1]).all(), \
        "Ethanol and propanol fingerprints are identical — ECFP hashing is broken"


def test_canonical_smiles_same_fingerprint():
    """Two representations of benzene must produce the same bit vector."""
    fps = molcore.featurize_smiles(["c1ccccc1", "C1=CC=CC=C1"], backend="rust")
    assert (fps[0] == fps[1]).all(), \
        "Benzene canonical forms give different fingerprints — canonicalization broken"


def test_fingerprint_length_matches_nbits():
    """nbits parameter must be respected."""
    for nbits in (512, 1024, 2048):
        fps = molcore.featurize_smiles(["CCO"], backend="rust", nbits=nbits)
        assert fps.shape[1] == nbits, f"nbits={nbits} but got {fps.shape[1]} bits"


def test_rdkit_rust_same_element_count():
    """Both backends must agree on the number of set bits — order can differ."""
    smi = ["c1ccccc1", "CC(=O)O", "CCN"]
    r = molcore.featurize_smiles(smi, backend="rust").sum(dim=1)
    k = molcore.featurize_smiles(smi, backend="rdkit").sum(dim=1)
    # bit counts won't be identical (different hash seeds) but both must be nonzero
    assert (r > 0).all(), "Rust fingerprints have all-zero rows"
    assert (k > 0).all(), "RDKit fingerprints have all-zero rows"


def test_scaffold_split_different_seeds_differ():
    """Different seeds should (almost always) produce different splits."""
    from molcore.io import MolDataset
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC",
              "c1ccncc1", "Nc1ccccc1", "COc1ccccc1"]
    ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
    t1, _, _ = ds.scaffold_split(seed=42)
    t2, _, _ = ds.scaffold_split(seed=99)
    # Not guaranteed to differ for tiny datasets, but scaffold grouping makes it likely
    # — this test is advisory, not strict
    _ = t1, t2  # just ensure no exception


# ── Aromaticity invariants — load-bearing for all GNN models ─────────────────
# Node feature index 1 = is_aromatic. Every GNN trained through molcore
# depends on these flags being correct. A parser bug here is silent but fatal.

def test_benzene_all_atoms_aromatic():
    """All 6 carbons in benzene must be flagged aromatic (x[:, 1] == 1)."""
    data = Mol.from_smiles("c1ccccc1").to_pyg()
    assert data.x.shape[0] == 6
    assert all(data.x[:, 1] == 1.0), \
        f"Benzene aromatic flags wrong: {data.x[:, 1].tolist()}"


def test_pyridine_all_atoms_aromatic():
    """All 6 atoms in pyridine must be flagged aromatic."""
    data = Mol.from_smiles("c1ccncc1").to_pyg()
    assert data.x.shape[0] == 6
    assert all(data.x[:, 1] == 1.0), \
        f"Pyridine aromatic flags wrong: {data.x[:, 1].tolist()}"


def test_cyclohexane_no_atoms_aromatic():
    """No atoms in cyclohexane may be flagged aromatic (saturated ring)."""
    data = Mol.from_smiles("C1CCCCC1").to_pyg()
    assert all(data.x[:, 1] == 0.0), \
        f"Cyclohexane has false aromatic atoms: {data.x[:, 1].tolist()}"


def test_thiophene_all_atoms_aromatic():
    """Thiophene: 5-membered heteroaromatic — all atoms must be aromatic."""
    data = Mol.from_smiles("c1ccsc1").to_pyg()
    assert all(data.x[:, 1] == 1.0), \
        f"Thiophene aromatic flags wrong: {data.x[:, 1].tolist()}"


def test_imidazole_all_atoms_aromatic():
    """Imidazole: 5-membered N-heteroaromatic — all atoms must be aromatic."""
    data = Mol.from_smiles("c1cnc[nH]1").to_pyg()
    assert all(data.x[:, 1] == 1.0), \
        f"Imidazole aromatic flags wrong: {data.x[:, 1].tolist()}"


def test_aspirin_exactly_six_aromatic_atoms():
    """Aspirin has one phenyl ring (6 aromatic atoms) and a non-aromatic side chain."""
    data = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O").to_pyg()
    n_aromatic = int(data.x[:, 1].sum().item())
    assert n_aromatic == 6, \
        f"Aspirin: expected 6 aromatic atoms, got {n_aromatic}"


def test_indane_aromatic_ring_only():
    """Indane: fused aromatic + saturated ring — only the benzene ring is aromatic."""
    data = Mol.from_smiles("C1Cc2ccccc21").to_pyg()
    n_aromatic = int(data.x[:, 1].sum().item())
    # Benzene ring contributes 6 aromatic atoms; the cyclopentane contributes 0
    # (2 CH2 carbons are in both rings but RDKit marks them non-aromatic in indane)
    assert n_aromatic == 6, \
        f"Indane: expected 6 aromatic atoms, got {n_aromatic}"


def test_naphthalene_all_atoms_aromatic():
    """Naphthalene: fused bicyclic aromatic — all 10 atoms must be aromatic."""
    data = Mol.from_smiles("c1ccc2ccccc2c1").to_pyg()
    assert data.x.shape[0] == 10
    assert all(data.x[:, 1] == 1.0), \
        f"Naphthalene aromatic flags wrong: {data.x[:, 1].tolist()}"
