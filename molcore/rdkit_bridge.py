"""
rdkit_bridge.py — ALL RDKit calls are isolated here.
One file to update when RDKit changes an API.
Never import rdkit anywhere else in the hot path.
"""
import itertools
import re

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
    _, uncharger, _ = _std_objects()
    return Chem.MolToSmiles(uncharger.uncharge(mol))


def strip_salts(smiles: str) -> str:
    mol = from_smiles(smiles)
    chooser, _, _ = _std_objects()
    return Chem.MolToSmiles(chooser.choose(mol))


def canonicalize(smiles: str) -> str:
    return Chem.MolToSmiles(from_smiles(smiles))


def mol_to_smiles(rdmol) -> str:
    """Convert an RDKit Mol object to canonical SMILES. Used by io.py to keep rdkit imports isolated."""
    return Chem.MolToSmiles(rdmol)


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


# ---------------------------------------------------------------------------
# SDF / Mol / Mol2 file I/O
# ---------------------------------------------------------------------------

def from_sdf_file(
    path: str,
    sanitize: bool = True,
    remove_hs: bool = True,
) -> list[tuple]:
    """
    Read an SDF (or gzipped .sdf.gz) file.
    Returns list of (rdmol, properties_dict). Invalid records are silently skipped.
    """
    import gzip as _gzip
    path = str(path)
    if path.endswith(".gz"):
        with _gzip.open(path) as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, sanitize=sanitize, removeHs=remove_hs)
            result = [(m, {k: m.GetProp(k) for k in m.GetPropNames()}) for m in suppl if m]
    else:
        suppl = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=remove_hs)
        result = [(m, {k: m.GetProp(k) for k in m.GetPropNames()}) for m in suppl if m]
    return result


def from_molblock(molblock: str, sanitize: bool = True):
    """Parse an MDL Mol block string. Raises ValueError on failure."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=sanitize, removeHs=True)
    if mol is None:
        raise ValueError("Could not parse Mol block")
    return mol


def write_sdf(
    smiles_list: list[str],
    path: str,
    properties: "dict[str, list] | None" = None,
) -> None:
    """
    Write SMILES to an SDF file. properties maps column name → list of values
    parallel to smiles_list. Invalid SMILES are written as empty records so
    row indices are preserved.
    """
    if properties:
        for key, vals in properties.items():
            if len(vals) != len(smiles_list):
                raise ValueError(
                    f"properties[{key!r}] has {len(vals)} values but smiles_list has "
                    f"{len(smiles_list)} — lengths must match"
                )
    writer = Chem.SDWriter(str(path))
    for i, smi in enumerate(smiles_list):
        try:
            mol = from_smiles(smi)
            if properties:
                for key, vals in properties.items():
                    mol.SetProp(str(key), str(vals[i]))
        except ValueError:
            mol = Chem.RWMol()
        writer.write(mol)
    writer.close()


def mol_to_sdf_block(smiles: str) -> str:
    """Return an SDF-formatted Mol block string for one SMILES."""
    from io import StringIO as _StringIO
    mol = from_smiles(smiles)
    sio = _StringIO()
    w = Chem.SDWriter(sio)
    w.write(mol)
    w.close()
    return sio.getvalue()


# ---------------------------------------------------------------------------
# Full descriptor set — named RDKit descriptors with presets
# ---------------------------------------------------------------------------

_LIPINSKI_NAMES: list[str] = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount",
]

_DRUGLIKE_NAMES: list[str] = _LIPINSKI_NAMES + [
    "FractionCSP3", "HeavyAtomCount", "NumAromaticRings",
    "NumAliphaticRings", "NumSaturatedRings", "BertzCT",
    "MolMR", "LabuteASA",
]

DESCRIPTOR_PRESETS: dict[str, list[str]] = {
    "lipinski": _LIPINSKI_NAMES,
    "druglike": _DRUGLIKE_NAMES,
}


def list_descriptor_names() -> list[str]:
    """Return all ~200 RDKit 2D descriptor names."""
    from rdkit.Chem import Descriptors as _D
    return [name for name, _ in _D.descList]


def calc_named_descriptors(
    smiles: list[str],
    names: "list[str] | None" = None,
    preset: "str | None" = None,
) -> "tuple[np.ndarray, list[str]]":
    """
    Compute named RDKit descriptors for a batch of SMILES.

    Exactly one of `names` or `preset` should be given.
    preset: "lipinski" | "druglike" | "all"
    Returns (array of shape (N, D) float32, list_of_names).
    Invalid SMILES rows are filled with NaN.
    """
    from rdkit.Chem import Descriptors as _D

    if preset == "all":
        col_names = list_descriptor_names()
    elif preset is not None:
        col_names = DESCRIPTOR_PRESETS.get(preset)
        if col_names is None:
            raise ValueError(
                f"Unknown preset {preset!r}. Choose from: {list(DESCRIPTOR_PRESETS)!r} or 'all'"
            )
    elif names is not None:
        col_names = list(names)
    else:
        col_names = _LIPINSKI_NAMES

    desc_fns: dict[str, object] = {name: fn for name, fn in _D.descList}
    unknown = [n for n in col_names if n not in desc_fns]
    if unknown:
        raise ValueError(f"Unknown descriptor names: {unknown!r}")

    n, d = len(smiles), len(col_names)
    out = np.full((n, d), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        try:
            mol = from_smiles(smi)
            for j, name in enumerate(col_names):
                try:
                    out[i, j] = float(desc_fns[name](mol))  # type: ignore[operator]
                except Exception:
                    pass
        except ValueError:
            pass
    return out, col_names


# ---------------------------------------------------------------------------
# Additional fingerprint types
# ---------------------------------------------------------------------------

def maccs_keys(smiles: list[str]) -> np.ndarray:
    """MACCS 166-bit keys. Returns (N, 167) uint8 (bit 0 unused, RDKit convention)."""
    from rdkit.Chem import MACCSkeys, DataStructs
    rows = []
    for smi in smiles:
        arr = np.zeros(167, dtype=np.uint8)
        try:
            fp = MACCSkeys.GenMACCSKeys(from_smiles(smi))
            DataStructs.ConvertToNumpyArray(fp, arr)
        except ValueError:
            pass
        rows.append(arr)
    return np.stack(rows)


def atom_pairs_fp(smiles: list[str], nbits: int = 2048) -> np.ndarray:
    """Atom-pair hashed fingerprints. Returns (N, nbits) uint8."""
    from rdkit.Chem import DataStructs
    from rdkit.Chem.AtomPairs import Pairs
    rows = []
    for smi in smiles:
        arr = np.zeros(nbits, dtype=np.uint8)
        try:
            fp = Pairs.GetHashedAtomPairFingerprintAsBitVect(from_smiles(smi), nBits=nbits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except (ValueError, Exception):
            pass
        rows.append(arr)
    return np.stack(rows)


def topological_torsions_fp(smiles: list[str], nbits: int = 2048) -> np.ndarray:
    """Topological torsion fingerprints. Returns (N, nbits) uint8."""
    from rdkit.Chem import DataStructs
    from rdkit.Chem.AtomPairs import Torsions
    rows = []
    for smi in smiles:
        arr = np.zeros(nbits, dtype=np.uint8)
        try:
            fp = Torsions.GetHashedTopologicalTorsionFingerprintAsBitVect(
                from_smiles(smi), nBits=nbits
            )
            DataStructs.ConvertToNumpyArray(fp, arr)
        except (ValueError, Exception):
            pass
        rows.append(arr)
    return np.stack(rows)


def rdkit_path_fp(smiles: list[str], nbits: int = 2048) -> np.ndarray:
    """RDKit path-based fingerprints. Returns (N, nbits) uint8."""
    from rdkit.Chem import DataStructs, RDKFingerprint
    rows = []
    for smi in smiles:
        arr = np.zeros(nbits, dtype=np.uint8)
        try:
            fp = RDKFingerprint(from_smiles(smi), fpSize=nbits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except (ValueError, Exception):
            pass
        rows.append(arr)
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Full standardization pipeline
# ---------------------------------------------------------------------------

def _std_objects():
    """Return (fragment_chooser, uncharger, tautomer_enumerator) — cached per-process."""
    if not hasattr(_std_objects, "_cache"):
        _std_objects._cache = (
            MolStandardize.rdMolStandardize.LargestFragmentChooser(),
            MolStandardize.rdMolStandardize.Uncharger(),
            MolStandardize.rdMolStandardize.TautomerEnumerator(),
        )
    return _std_objects._cache


def standardize(smiles: str) -> str:
    """
    Full MolVS-style standardization in one call:
      1. Keep largest fragment (strip counterions/salts)
      2. Neutralize charges
      3. Canonical tautomer
      4. Return canonical SMILES

    Raises ValueError for unparseable SMILES.
    Standardizer objects are cached at the module level — no per-call overhead.
    """
    frag, uncharge, taut = _std_objects()
    mol = from_smiles(smiles)
    mol = frag.choose(mol)
    mol = uncharge.uncharge(mol)
    mol = taut.Canonicalize(mol)
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Maximum Common Substructure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Butina clustering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diversity picking (MaxMin algorithm)
# ---------------------------------------------------------------------------

def diversity_pick(
    smiles_list: list[str],
    n: int,
    nbits: int = 2048,
    radius: int = 2,
    seed: int = 0,
) -> list[int]:
    """
    Select `n` maximally diverse molecules using the MaxMin algorithm.

    Starts with the molecule closest to the centroid (or the molecule at
    position `seed` index if seed is an int < len(smiles_list)), then
    iteratively picks the molecule with the highest minimum Tanimoto
    distance to all already-selected molecules.

    Returns a list of `n` indices into `smiles_list`.
    Invalid SMILES are excluded from selection (their indices are never returned).

    Time complexity: O(n × N) fingerprint comparisons, where N = len(smiles_list).
    For N ≤ 100k and n ≤ 1k, this runs in seconds on a single core.

    Scaling note: at N > 500k the O(n × N) cost becomes the bottleneck. For very
    large libraries, pre-cluster with butina_cluster() at a loose cutoff (~0.8),
    then run diversity_pick() within each cluster — this reduces the effective N
    to cluster size. A built-in fast=True mode for this pattern is planned for v0.3.
    """
    from rdkit.Chem import DataStructs

    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = from_smiles(smi)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
            valid_idx.append(i)
        except ValueError:
            pass

    n_valid = len(fps)
    if n_valid == 0:
        return []
    n = min(n, n_valid)

    # Seed: first valid molecule (index 0 in fps)
    seed_pos = seed % n_valid
    selected_pos = [seed_pos]
    # min_dist[i] = minimum Tanimoto distance from fps[i] to any selected fp
    min_dist = [1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[seed_pos], fps)]

    for _ in range(n - 1):
        # Pick the molecule with the highest minimum distance to selected set
        next_pos = max(
            (i for i in range(n_valid) if i not in selected_pos),
            key=lambda i: min_dist[i],
        )
        selected_pos.append(next_pos)
        # Update min_dist with distances to the newly selected molecule
        new_sims = DataStructs.BulkTanimotoSimilarity(fps[next_pos], fps)
        for i, sim in enumerate(new_sims):
            d = 1.0 - sim
            if d < min_dist[i]:
                min_dist[i] = d

    return [valid_idx[p] for p in selected_pos]


def butina_cluster(
    smiles_list: list[str],
    cutoff: float = 0.4,
    nbits: int = 2048,
    radius: int = 2,
) -> list[int]:
    """
    Cluster molecules using the Butina algorithm on Tanimoto distance.

    cutoff: Tanimoto *distance* threshold (= 1 - similarity). Default 0.4
            means molecules with similarity ≥ 0.6 end up in the same cluster.

    Returns a list of integer cluster IDs (0-indexed), one per input SMILES.
    Invalid SMILES receive cluster ID -1.
    Cluster 0 is always the largest cluster.
    """
    from rdkit.ML.Cluster import Butina
    from rdkit.Chem import DataStructs

    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = from_smiles(smi)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
            valid_idx.append(i)
        except ValueError:
            pass

    cluster_ids = [-1] * len(smiles_list)
    n = len(fps)
    if n == 0:
        return cluster_ids

    # Lower-triangle Tanimoto distance matrix required by Butina
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    for cluster_id, members in enumerate(clusters):
        for pos in members:
            cluster_ids[valid_idx[pos]] = cluster_id

    return cluster_ids


# ---------------------------------------------------------------------------
# Matched Molecular Pair Analysis (MMPA)
# ---------------------------------------------------------------------------

# Matches any single acyclic bond, excluding bridgehead N/O/S in rings
_ACYCLIC_SINGLE = Chem.MolFromSmarts(
    "[!$([#7;X3;r])&!$([#8;X2;r])&!$([#16;X2;r])]-&!@[*]"
)


def _mmpa_normalize_attachment(smi: str) -> str:
    """Normalize all atom map numbers in an end-fragment SMILES to :1."""
    return re.sub(r":\d+\]", ":1]", smi)


def _mmpa_canonical_linker(smi: str) -> str:
    """Return the lex-min of a linker SMILES and its :1↔:2 swap.

    Ensures the same linker produces the same string regardless of which
    cut was labelled bond-1 vs bond-2.
    """
    alt = smi.replace(":1]", ":99x]").replace(":2]", ":1]").replace(":99x]", ":2]")
    return min(smi, alt)


def _mmpa_environment_smarts(mol, atom_idx: int, radius: int) -> str:
    """
    Return a SMARTS string encoding the chemical environment around `atom_idx`
    up to `radius` bonds away.  radius=0 returns '' (no context).

    Used to annotate each MMP pair with the local environment at the
    attachment point so downstream analysis can filter by context.
    """
    if radius == 0:
        return ""
    try:
        from rdkit.Chem import AllChem
        env = AllChem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        amap: dict[int, int] = {}
        submol = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=[mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in env]
            + [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in env],
            bondsToUse=list(env),
            atomSymbols=None,
            atomCounts=amap,
            isomericSmiles=False,
        )
        return submol or ""
    except Exception:
        return ""


def _mmpa_single_cut(valid_smiles: list[str], radius: int = 0) -> "list[dict]":
    from collections import defaultdict

    core_to_entries: dict[str, dict[str, str]] = defaultdict(dict)
    core_to_env: dict[str, str] = {}

    for smi in valid_smiles:
        try:
            mol = from_smiles(smi)
        except ValueError:
            continue

        matches = mol.GetSubstructMatches(_ACYCLIC_SINGLE)
        seen_cores: set[str] = set()

        for bond_match in matches:
            bi, bj = bond_match
            bond = mol.GetBondBetweenAtoms(bi, bj)
            if bond is None or bond.IsInRing():
                continue

            em = Chem.RWMol(mol)
            em.RemoveBond(bi, bj)
            em.GetAtomWithIdx(bi).SetAtomMapNum(1)
            em.GetAtomWithIdx(bj).SetAtomMapNum(1)

            try:
                frags = Chem.MolToSmiles(em.GetMol()).split(".")
            except Exception:
                continue

            if len(frags) != 2:
                continue

            frags.sort(key=lambda s: s.count("*") + len(s), reverse=True)
            core_smi, sub_smi = frags[0], frags[1]

            try:
                core_can = Chem.MolToSmiles(Chem.MolFromSmiles(core_smi))
                sub_can  = Chem.MolToSmiles(Chem.MolFromSmiles(sub_smi))
            except Exception:
                continue

            if core_can in seen_cores:
                continue
            seen_cores.add(core_can)
            core_to_entries[core_can][smi] = sub_can
            if radius > 0 and core_can not in core_to_env:
                core_to_env[core_can] = _mmpa_environment_smarts(mol, bi, radius)

    pairs = []
    for core_smi, mol_to_sub in core_to_entries.items():
        entries = sorted(mol_to_sub.items())
        env = core_to_env.get(core_smi, "")
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mol_a, sub_a = entries[i]
                mol_b, sub_b = entries[j]
                if sub_a == sub_b:
                    continue
                pair: dict = {
                    "mol_a":     mol_a,
                    "mol_b":     mol_b,
                    "smiles_a":  sub_a,
                    "smiles_b":  sub_b,
                    "core":      core_smi,
                    "transform": f"{sub_a}>>{sub_b}",
                }
                if radius > 0:
                    pair["environment"] = env
                pairs.append(pair)
    return pairs


def _mmpa_double_cut(valid_smiles: list[str]) -> "list[dict]":
    """Double-cut MMPA: find pairs of molecules differing only in a linker fragment.

    Algorithm:
      1. For each molecule, enumerate all pairs of non-adjacent acyclic bonds.
      2. Cut both bonds → expect exactly 3 fragments (two terminal ends + one linker).
      3. The linker is the fragment that carries both cut-point markers.
      4. Group molecules by frozenset({end1_canonical, end2_canonical}).
      5. Within each group, emit pairs that differ in the linker.
    """
    from collections import defaultdict

    # frozenset({end1_can, end2_can}) → {orig_smiles → linker_can}
    core_to_entries: defaultdict = defaultdict(dict)

    for smi in valid_smiles:
        try:
            mol = from_smiles(smi)
        except ValueError:
            continue

        matches = mol.GetSubstructMatches(_ACYCLIC_SINGLE)
        # Deduplicate bonds as canonical (min_idx, max_idx) pairs
        bonds = list({(min(m[0], m[1]), max(m[0], m[1])) for m in matches})

        seen_cores: set = set()

        for (ai, aj), (bi, bj) in itertools.combinations(bonds, 2):
            # Skip adjacent bonds (share an atom) — produce degenerate fragments
            if len({ai, aj, bi, bj}) < 4:
                continue

            em = Chem.RWMol(mol)
            try:
                em.RemoveBond(ai, aj)
                em.RemoveBond(bi, bj)
            except Exception:
                continue

            # Label: first-cut atoms → :1, second-cut atoms → :2
            em.GetAtomWithIdx(ai).SetAtomMapNum(1)
            em.GetAtomWithIdx(aj).SetAtomMapNum(1)
            em.GetAtomWithIdx(bi).SetAtomMapNum(2)
            em.GetAtomWithIdx(bj).SetAtomMapNum(2)

            try:
                frags_smi = Chem.MolToSmiles(em.GetMol()).split(".")
            except Exception:
                continue

            if len(frags_smi) != 3:
                continue

            # Linker = the fragment that carries both :1] and :2] markers
            linker_idx = next(
                (i for i, f in enumerate(frags_smi) if ":1]" in f and ":2]" in f),
                None,
            )
            if linker_idx is None:
                continue

            end_idxs = [i for i in range(3) if i != linker_idx]
            end1_raw   = frags_smi[end_idxs[0]]
            end2_raw   = frags_smi[end_idxs[1]]
            linker_raw = frags_smi[linker_idx]

            try:
                end1_can = Chem.MolToSmiles(
                    Chem.MolFromSmiles(_mmpa_normalize_attachment(end1_raw))
                )
                end2_can = Chem.MolToSmiles(
                    Chem.MolFromSmiles(_mmpa_normalize_attachment(end2_raw))
                )
                linker_can = _mmpa_canonical_linker(
                    Chem.MolToSmiles(Chem.MolFromSmiles(linker_raw))
                )
            except Exception:
                continue

            core_key = frozenset({end1_can, end2_can})
            if core_key in seen_cores:
                continue
            seen_cores.add(core_key)
            core_to_entries[core_key][smi] = linker_can

    pairs = []
    for core_key, mol_to_linker in core_to_entries.items():
        entries = sorted(mol_to_linker.items())
        core_display = ".".join(sorted(core_key))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                mol_a, linker_a = entries[i]
                mol_b, linker_b = entries[j]
                if linker_a == linker_b:
                    continue
                pairs.append({
                    "mol_a":     mol_a,
                    "mol_b":     mol_b,
                    "smiles_a":  linker_a,
                    "smiles_b":  linker_b,
                    "core":      core_display,
                    "transform": f"{linker_a}>>{linker_b}",
                })
    return pairs


def mmpa(
    smiles_list: list[str],
    max_cut_bonds: int = 1,
    radius: int = 3,
) -> "list[dict]":
    """
    Find all Matched Molecular Pairs (MMPs) in a list of molecules.

    Single-cut (max_cut_bonds=1):
      Molecules differing by one substituent at a single attachment point.
      Core = large fragment; variable = small substituent fragment.

    Double-cut (max_cut_bonds=2):
      Molecules differing by a linker between two constant terminal fragments.
      Core = the two terminal ends; variable = the linker between them.
      Useful for bioisostere linker replacement and scaffold-hopping SAR.

    Args:
        smiles_list   : input SMILES (duplicates and invalids are silently skipped)
        max_cut_bonds : 1 (single-cut, default) or 2 (double-cut)
        radius        : heavy-atom radius around the attachment point used to
                        compute a local environment SMARTS. 0 = no environment
                        (default, backward-compatible). When >0, each pair dict
                        gains an ``"environment"`` key with the SMARTS string.

    Returns list of dicts with keys:
        mol_a, mol_b   : SMILES of the two paired molecules
        smiles_a       : variable fragment from mol_a (substituent or linker)
        smiles_b       : variable fragment from mol_b
        core           : constant part SMILES
        transform      : 'smiles_a>>smiles_b'

    Pairs are emitted in canonical order (mol_a < mol_b lexicographically).
    """
    if max_cut_bonds not in (1, 2):
        raise ValueError(
            f"max_cut_bonds must be 1 or 2, got {max_cut_bonds}"
        )

    # Canonicalize and deduplicate
    seen: set[str] = set()
    valid: list[str] = []
    for smi in smiles_list:
        try:
            canon = Chem.MolToSmiles(from_smiles(smi))
        except ValueError:
            continue
        if canon not in seen:
            seen.add(canon)
            valid.append(canon)

    if max_cut_bonds == 1:
        return _mmpa_single_cut(valid, radius=radius)
    return _mmpa_double_cut(valid)


def find_mcs(
    smiles_list: list[str],
    timeout: int = 5,
    complete_rings_only: bool = False,
    match_valences: bool = False,
) -> str:
    """
    Find the Maximum Common Substructure of a list of molecules.
    Returns SMARTS string (empty string if MCS is trivial or times out).
    Raises ValueError if fewer than 2 valid molecules are given.
    """
    from rdkit.Chem import rdFMCS
    mols = []
    for smi in smiles_list:
        try:
            mols.append(from_smiles(smi))
        except ValueError:
            pass
    if len(mols) < 2:
        raise ValueError("Need at least 2 valid molecules for MCS")
    result = rdFMCS.FindMCS(
        mols,
        timeout=timeout,
        completeRingsOnly=complete_rings_only,
        matchValences=match_valences,
        ringMatchesRingOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    return result.smartsString if result.numAtoms > 0 else ""


# ---------------------------------------------------------------------------
# R-Group decomposition
# ---------------------------------------------------------------------------

def rgroup_decompose(
    core_smiles: str,
    smiles_list: list[str],
) -> list[dict]:
    """
    Decompose molecules into a core + R-groups using RDKit rdRGroupDecomposition.

    core_smiles: SMARTS or SMILES of the core scaffold (SMARTS preferred for wildcards).
    Returns list[dict] — one dict per input molecule with keys 'Core', 'R1', 'R2', ...
    Non-matching molecules get an empty dict.
    """
    from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose

    core = Chem.MolFromSmarts(core_smiles) or from_smiles(core_smiles)
    mols, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mols.append(from_smiles(smi))
            valid_idx.append(i)
        except ValueError:
            pass

    rows_out: list[dict] = [{} for _ in smiles_list]
    if not mols:
        return rows_out

    groups, _ = RGroupDecompose([core], mols, asSmiles=True, asRows=True)
    for j, group in enumerate(groups):
        rows_out[valid_idx[j]] = group
    return rows_out


# ---------------------------------------------------------------------------
# 2D depiction
# ---------------------------------------------------------------------------

def mol_to_svg(
    smiles: str,
    width: int = 300,
    height: int = 200,
    highlight_atoms: "list[int] | None" = None,
    highlight_bonds: "list[int] | None" = None,
) -> str:
    """Render a molecule to an SVG string using RDKit 2D coordinates."""
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = from_smiles(smiles)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightBonds=highlight_bonds or [],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_png(
    smiles: str,
    path: str,
    width: int = 300,
    height: int = 200,
) -> None:
    """
    Render a molecule to a PNG file.
    Requires RDKit built with Cairo support. Falls back to a helpful error if unavailable.
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = from_smiles(smiles)
    AllChem.Compute2DCoords(mol)
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    except Exception as exc:
        raise RuntimeError(
            "PNG rendering requires RDKit with Cairo support. "
            "Use mol_to_svg() for a dependency-free alternative."
        ) from exc
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    with open(str(path), "wb") as fh:
        fh.write(drawer.GetDrawingText())


def mols_to_grid_svg(
    smiles_list: list[str],
    mols_per_row: int = 4,
    sub_img_size: "tuple[int, int]" = (200, 150),
    legends: "list[str] | None" = None,
) -> str:
    """Render a grid of molecules as a single SVG string."""
    from rdkit.Chem import Draw
    mols, valid_legends = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = from_smiles(smi)
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            valid_legends.append(legends[i] if legends else smi)
        except ValueError:
            pass
    if not mols:
        return "<svg/>"
    try:
        return Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=valid_legends,
            useSVG=True,
        )
    except TypeError:
        from io import BytesIO as _BytesIO
        import base64 as _b64
        img = Draw.MolsToGridImage(
            mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=valid_legends
        )
        buf = _BytesIO()
        img.save(buf, format="PNG")
        b64 = _b64.b64encode(buf.getvalue()).decode()
        w, h = img.size
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/></svg>'
        )


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
