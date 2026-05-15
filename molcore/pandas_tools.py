"""
molcore.pandas_tools — pandas-first API for computational chemists.

Most RDKit users think in DataFrames. This module meets them there:

    import molcore.pandas_tools as mpt

    df = mpt.load_sdf("library.sdf")           # → DataFrame with 'Mol' column
    df = mpt.add_mol_column(df, "SMILES")       # SMILES col → Mol objects
    mpt.add_fingerprints(df, inplace=True)      # adds 'fp' column (torch.Tensor)
    mpt.add_descriptors(df, preset="lipinski")  # adds descriptor columns
    hits = mpt.filter_by_smarts(df, "c1ccncc1") # substructure filter
    mpt.write_sdf(df, "filtered.sdf")           # write back to SDF
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_sdf(
    path: str,
    mol_col: str = "Mol",
    sanitize: bool = True,
    remove_hs: bool = True,
) -> "pd.DataFrame":
    """
    Read an SDF (or .sdf.gz) file into a DataFrame.

    Each molecule becomes a row. SD properties become columns.
    A 'smiles' column and a `mol_col` column (holding molcore Mol objects) are added.
    Invalid records are silently skipped.
    """
    import pandas as pd
    from molcore.rdkit_bridge import from_sdf_file, canonicalize
    from molcore.molecule import Mol
    from rdkit import Chem as _Chem

    records = from_sdf_file(path, sanitize=sanitize, remove_hs=remove_hs)
    rows = []
    for rdmol, props in records:
        try:
            smi = canonicalize(_Chem.MolToSmiles(rdmol))
            row = dict(props)
            row["smiles"] = smi
            row[mol_col] = Mol.from_smiles(smi)
            rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["smiles", mol_col])


def write_sdf(
    df: "pd.DataFrame",
    path: str,
    smiles_col: str = "smiles",
    extra_cols: "list[str] | None" = None,
) -> None:
    """
    Write a DataFrame to an SDF file.

    smiles_col provides the structures. All other string/numeric columns
    (or those listed in extra_cols) are written as SD properties.
    """
    from molcore.rdkit_bridge import write_sdf as _write_sdf
    smiles = df[smiles_col].tolist()
    skip = {smiles_col, "Mol", "mol", "rdmol"}
    if extra_cols is not None:
        props = {c: df[c].tolist() for c in extra_cols if c in df.columns}
    else:
        props = {
            c: df[c].tolist()
            for c in df.columns
            if c not in skip and df[c].dtype.kind in "ifUSO"
        }
    _write_sdf(smiles, path, properties=props or None)


# ---------------------------------------------------------------------------
# Mol column helpers
# ---------------------------------------------------------------------------

def add_mol_column(
    df: "pd.DataFrame",
    smiles_col: str = "smiles",
    mol_col: str = "Mol",
    errors: str = "coerce",
) -> "pd.DataFrame":
    """
    Add a 'Mol' column to a DataFrame by parsing a SMILES column.

    errors='coerce'  → invalid SMILES become None
    errors='raise'   → invalid SMILES raise ValueError
    Returns a new DataFrame (original unchanged).
    """
    from molcore.molecule import Mol
    import pandas as pd

    def _parse(smi: str):
        try:
            return Mol.from_smiles(str(smi))
        except Exception:
            if errors == "raise":
                raise
            return None

    out = df.copy()
    out[mol_col] = df[smiles_col].apply(_parse)
    return out


def add_smiles_column(
    df: "pd.DataFrame",
    mol_col: str = "Mol",
    smiles_col: str = "smiles",
) -> "pd.DataFrame":
    """Extract canonical SMILES from a Mol column into a new string column."""
    out = df.copy()
    out[smiles_col] = df[mol_col].apply(
        lambda m: m.smiles if m is not None else None
    )
    return out


# ---------------------------------------------------------------------------
# Fingerprints & descriptors
# ---------------------------------------------------------------------------

def add_fingerprints(
    df: "pd.DataFrame",
    smiles_col: str = "smiles",
    kind: str = "ecfp4",
    nbits: int = 2048,
    radius: int = 2,
    backend: str = "rust",
    col_prefix: str = "fp",
    expand_cols: bool = False,
    inplace: bool = False,
) -> "pd.DataFrame":
    """
    Compute fingerprints for each molecule and add them to the DataFrame.

    kind: 'ecfp4' | 'maccs' | 'atom_pairs' | 'topological_torsions' | 'rdkit'
    expand_cols=False → single 'fp' column holding a numpy array per row
    expand_cols=True  → one column per bit (fp_0, fp_1, ...)
    """
    import numpy as np
    import pandas as pd
    from molcore.pipeline import featurize_smiles
    from molcore.rdkit_bridge import (
        maccs_keys, atom_pairs_fp, topological_torsions_fp, rdkit_path_fp,
    )

    smiles = df[smiles_col].tolist()
    kind_lc = kind.lower()
    if kind_lc == "ecfp4":
        arr = featurize_smiles(smiles, backend=backend, radius=radius, nbits=nbits).numpy()
    elif kind_lc == "maccs":
        arr = maccs_keys(smiles)
    elif kind_lc in ("atom_pairs", "atompairs"):
        arr = atom_pairs_fp(smiles, nbits=nbits)
    elif kind_lc in ("topological_torsions", "torsions"):
        arr = topological_torsions_fp(smiles, nbits=nbits)
    elif kind_lc == "rdkit":
        arr = rdkit_path_fp(smiles, nbits=nbits)
    else:
        raise ValueError(f"Unknown fingerprint kind: {kind!r}")

    out = df if inplace else df.copy()
    if expand_cols:
        import pandas as _pd
        fp_df = _pd.DataFrame(arr, columns=[f"{col_prefix}_{i}" for i in range(arr.shape[1])], index=out.index)
        out = _pd.concat([out, fp_df], axis=1)
    else:
        out[col_prefix] = list(arr)
    return out


def add_descriptors(
    df: "pd.DataFrame",
    smiles_col: str = "smiles",
    names: "list[str] | None" = None,
    preset: "str | None" = "lipinski",
    inplace: bool = False,
) -> "pd.DataFrame":
    """
    Compute RDKit descriptors and add them as columns to the DataFrame.

    preset: 'lipinski' | 'druglike' | 'all'  (ignored if `names` is given)
    Descriptor columns are named by the RDKit descriptor name (e.g. 'MolWt', 'TPSA').
    Invalid SMILES rows receive NaN values.
    """
    from molcore.rdkit_bridge import calc_named_descriptors
    smiles = df[smiles_col].tolist()
    arr, col_names = calc_named_descriptors(smiles, names=names, preset=preset)
    out = df if inplace else df.copy()
    for j, name in enumerate(col_names):
        out[name] = arr[:, j]
    return out


# ---------------------------------------------------------------------------
# Substructure filtering
# ---------------------------------------------------------------------------

def filter_by_smarts(
    df: "pd.DataFrame",
    smarts: str,
    smiles_col: str = "smiles",
    invert: bool = False,
) -> "pd.DataFrame":
    """
    Return rows whose molecule matches (or doesn't match) a SMARTS pattern.

    invert=True keeps molecules that do NOT match.
    Invalid SMILES are excluded from both branches.
    """
    from molcore.rdkit_bridge import filter_by_smarts as _filter
    hits = set(_filter(df[smiles_col].tolist(), smarts, invert=invert))
    mask = df[smiles_col].isin(hits)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scaffold operations
# ---------------------------------------------------------------------------

def add_scaffold_column(
    df: "pd.DataFrame",
    smiles_col: str = "smiles",
    scaffold_col: str = "scaffold",
    generic: bool = False,
) -> "pd.DataFrame":
    """Add a Murcko scaffold SMILES column."""
    from molcore.rdkit_bridge import murcko_scaffold
    out = df.copy()
    def _scaffold(smi: str) -> str:
        try:
            return murcko_scaffold(str(smi), generic=generic)
        except Exception:
            return ""
    out[scaffold_col] = df[smiles_col].apply(_scaffold)
    return out


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize_smiles(
    df: "pd.DataFrame",
    smiles_col: str = "smiles",
    inplace: bool = False,
) -> "pd.DataFrame":
    """
    Apply full MolVS-style standardization to a SMILES column in-place or in a copy.
    Invalid or unstandardizable SMILES are left unchanged.
    """
    from molcore.rdkit_bridge import standardize
    out = df if inplace else df.copy()
    def _std(smi: str) -> str:
        try:
            return standardize(str(smi))
        except Exception:
            return smi
    out[smiles_col] = df[smiles_col].apply(_std)
    return out
