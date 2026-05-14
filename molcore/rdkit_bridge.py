"""
rdkit_bridge.py — ALL RDKit calls are isolated here.
One file to update when RDKit changes an API.
Never import rdkit anywhere else in the hot path.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize


def from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def neutralize(smiles: str) -> str:
    mol = from_smiles(smiles)
    uncharger = MolStandardize.rdMolStandardize.Uncharger()
    return Chem.MolToSmiles(uncharger.uncharge(mol))


def strip_salts(smiles: str) -> str:
    mol = from_smiles(smiles)
    chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()
    return Chem.MolToSmiles(chooser.choose(mol))


def canonicalize(smiles: str) -> str:
    return Chem.MolToSmiles(from_smiles(smiles))


def ecfp4_rdkit(smiles: list[str], radius: int = 2, nbits: int = 2048) -> torch.Tensor:
    """RDKit-backend fingerprints — bit-identical to legacy trained models."""
    from rdkit.Chem import DataStructs
    rows = []
    for smi in smiles:
        mol = from_smiles(smi)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return torch.from_numpy(np.stack(rows))


def calc_descriptors_rdkit(smiles: list[str]) -> torch.Tensor:
    """RDKit MW, LogP, TPSA — exact values."""
    from rdkit.Chem import Descriptors
    rows = []
    for smi in smiles:
        mol = from_smiles(smi)
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
        ])
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# SMARTS substructure search
# ---------------------------------------------------------------------------

def _compile_smarts(smarts: str):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts!r}")
    return patt


def substructure_match(smiles: str, smarts: str) -> bool:
    """Return True if the molecule contains the SMARTS substructure."""
    return from_smiles(smiles).HasSubstructMatch(_compile_smarts(smarts))


def substructure_matches(smiles: str, smarts: str) -> list[tuple[int, ...]]:
    """Return all matching atom-index tuples for the SMARTS pattern."""
    return list(from_smiles(smiles).GetSubstructMatches(_compile_smarts(smarts)))


def filter_by_smarts(
    smiles_list: list[str],
    smarts: str,
    invert: bool = False,
) -> list[str]:
    """
    Return SMILES that match the SMARTS pattern.
    invert=True returns those that do NOT match (e.g. remove reactive groups).
    Invalid SMILES are silently skipped.
    """
    patt = _compile_smarts(smarts)
    result = []
    for smi in smiles_list:
        try:
            hit = from_smiles(smi).HasSubstructMatch(patt)
            if hit != invert:
                result.append(smi)
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Murcko scaffold decomposition
# ---------------------------------------------------------------------------

def murcko_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Return the Murcko scaffold of a molecule as canonical SMILES.

    generic=True replaces all atoms with C and all bonds with single bonds
    (framework scaffold — useful for scaffold-based clustering).
    Returns empty string if the molecule has no ring system.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = from_smiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(
    smiles_list: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Scaffold-based train/val/test split (no scaffold overlap between splits).

    Groups molecules by Murcko scaffold, then assigns groups to splits in
    descending scaffold-size order (largest scaffolds to train first).
    Returns (train, val, test) lists of SMILES.
    """
    import random as _random
    from collections import defaultdict

    rng = _random.Random(seed)
    scaffold_map: dict[str, list[str]] = defaultdict(list)

    for smi in smiles_list:
        try:
            sc = murcko_scaffold(smi)
        except ValueError:
            sc = smi  # ungrouped — treat as its own scaffold
        scaffold_map[sc].append(smi)

    # Sort by scaffold size descending for reproducibility, then shuffle within size-ties
    groups = sorted(scaffold_map.values(), key=len, reverse=True)

    n = len(smiles_list)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train, val, test = [], [], []
    for group in groups:
        if len(train) < n_train:
            train.extend(group)
        elif len(val) < n_val:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test


# ---------------------------------------------------------------------------
# Conformer generation + 3D descriptors
# ---------------------------------------------------------------------------

def generate_conformers(
    smiles: str,
    n_confs: int = 1,
    seed: int = 42,
    force_field: str = "MMFF94",
) -> list[np.ndarray]:
    """
    Generate 3D conformers using RDKit ETKDG + force-field optimization.

    Returns a list of (n_heavy_atoms, 3) float64 numpy arrays — one per conformer.
    Hydrogens are added for embedding then stripped from the returned coordinates.

    force_field: 'MMFF94' | 'MMFF94s' | 'UFF'
    Raises ValueError if embedding fails for all conformers.
    """
    from rdkit.Chem import rdDistGeom

    mol = from_smiles(smiles)
    mol_h = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0  # use all available

    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    if not conf_ids:
        raise ValueError(f"Conformer embedding failed for SMILES: {smiles!r}")

    if force_field.upper().startswith("MMFF"):
        AllChem.MMFFOptimizeMoleculeConfs(
            mol_h, mmffVariant=force_field, numThreads=0
        )
    else:
        AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=0)

    mol_no_h = Chem.RemoveHs(mol_h)
    confs = []
    for cid in conf_ids:
        conf = mol_no_h.GetConformer(cid)
        positions = np.array(conf.GetPositions(), dtype=np.float64)
        confs.append(positions)
    return confs


# ---------------------------------------------------------------------------
# Reaction transforms
# ---------------------------------------------------------------------------

def _compile_reaction(rxn_smarts: str):
    from rdkit.Chem import AllChem as _AC
    try:
        rxn = _AC.ReactionFromSmarts(rxn_smarts)
    except Exception as e:
        raise ValueError(f"Invalid reaction SMARTS: {rxn_smarts!r} — {e}") from e
    if rxn is None:
        raise ValueError(f"Invalid reaction SMARTS: {rxn_smarts!r}")
    return rxn


def react(smiles: str, rxn_smarts: str) -> list[str]:
    """
    Apply a reaction SMARTS to a single molecule (used as the first reactant).

    Returns a deduplicated list of unique product SMILES.
    Empty list if no products are formed.

    Example — ester hydrolysis:
        react("CC(=O)OCC", "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]")
    """
    rxn = _compile_reaction(rxn_smarts)
    mol = from_smiles(smiles)
    products = set()
    for product_set in rxn.RunReactants((mol,)):
        for p in product_set:
            try:
                Chem.SanitizeMol(p)
                smi = Chem.MolToSmiles(p)
                if smi:
                    products.add(smi)
            except Exception:
                pass
    return sorted(products)


def react_bimolecular(
    smiles_a: str,
    smiles_b: str,
    rxn_smarts: str,
) -> list[str]:
    """
    Apply a bimolecular reaction SMARTS.

    `smiles_a` is reactant 1, `smiles_b` is reactant 2.
    Returns deduplicated product SMILES.
    """
    rxn  = _compile_reaction(rxn_smarts)
    mol_a = from_smiles(smiles_a)
    mol_b = from_smiles(smiles_b)
    products = set()
    for product_set in rxn.RunReactants((mol_a, mol_b)):
        for p in product_set:
            try:
                Chem.SanitizeMol(p)
                smi = Chem.MolToSmiles(p)
                if smi:
                    products.add(smi)
            except Exception:
                pass
    return sorted(products)


def enumerate_reactions(
    reactants: list[str],
    rxn_smarts: str,
    max_products: int = 1000,
) -> list[str]:
    """
    Apply a unimolecular reaction SMARTS to every molecule in `reactants`.

    Returns a flat deduplicated list of all product SMILES, up to `max_products`.
    Molecules that don't react are silently skipped.
    """
    rxn = _compile_reaction(rxn_smarts)
    products: set[str] = set()
    for smi in reactants:
        if len(products) >= max_products:
            break
        try:
            mol = from_smiles(smi)
            for product_set in rxn.RunReactants((mol,)):
                for p in product_set:
                    try:
                        Chem.SanitizeMol(p)
                        s = Chem.MolToSmiles(p)
                        if s:
                            products.add(s)
                    except Exception:
                        pass
        except ValueError:
            pass
    return sorted(products)[:max_products]


def calc_descriptors_3d(smiles: str, seed: int = 42) -> dict[str, float]:
    """
    Compute shape descriptors that require a 3D conformer.

    Generates one ETKDG conformer internally.
    Returns: PMI1/2/3, asphericity, eccentricity, NPR1/2,
             radius_of_gyration, inertial_shape_factor, spherocity_index.
    """
    from rdkit.Chem import Descriptors3D, rdDistGeom

    mol = from_smiles(smiles)
    mol_h = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) < 0:
        raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol_h)

    return {
        "pmi1":                 Descriptors3D.PMI1(mol_h),
        "pmi2":                 Descriptors3D.PMI2(mol_h),
        "pmi3":                 Descriptors3D.PMI3(mol_h),
        "asphericity":          Descriptors3D.Asphericity(mol_h),
        "eccentricity":         Descriptors3D.Eccentricity(mol_h),
        "npr1":                 Descriptors3D.NPR1(mol_h),
        "npr2":                 Descriptors3D.NPR2(mol_h),
        "radius_of_gyration":   Descriptors3D.RadiusOfGyration(mol_h),
        "inertial_shape_factor": Descriptors3D.InertialShapeFactor(mol_h),
        "spherocity_index":     Descriptors3D.SpherocityIndex(mol_h),
    }
