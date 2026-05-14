"""
Tool: smarts — SMARTS substructure search and filtering.
Strict function signature for agent tool-use.
"""
from __future__ import annotations
from molcore.rdkit_bridge import filter_by_smarts, substructure_matches


def run(
    smiles: list[str],
    smarts: str,
    mode: str = "filter",
    invert: bool = False,
) -> dict:
    """
    Args:
        smiles : list of SMILES to search
        smarts : SMARTS pattern
        mode   : "filter"  → return matching SMILES list
                 "matches" → return per-molecule atom-index tuples
        invert : if True, return molecules that do NOT match (default False)

    Returns dict with:
        "hits"      : matching SMILES (mode="filter")
        "n_hits"    : count
        "all_matches": per-mol atom-index tuples (mode="matches")
    """
    if mode == "filter":
        hits = filter_by_smarts(smiles, smarts, invert=invert)
        return {"hits": hits, "n_hits": len(hits), "n_screened": len(smiles)}

    elif mode == "matches":
        result = {}
        for smi in smiles:
            try:
                result[smi] = substructure_matches(smi, smarts)
            except ValueError:
                result[smi] = []
        return {"all_matches": result, "n_screened": len(smiles)}

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'filter' or 'matches'.")
