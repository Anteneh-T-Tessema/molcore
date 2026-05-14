"""
Tool: scaffold — Murcko scaffold decomposition and scaffold-aware splitting.
Strict function signature for agent tool-use.
"""
from __future__ import annotations
from molcore.rdkit_bridge import murcko_scaffold, scaffold_split


def run(
    smiles: list[str],
    mode: str = "scaffold",
    generic: bool = False,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Args:
        smiles     : list of SMILES
        mode       : "scaffold"  → return Murcko scaffold per molecule
                     "split"     → scaffold-aware train/val/test split
                     "cluster"   → group molecules by scaffold
        generic    : use generic (carbon-only) scaffolds (default False)
        train_frac : fraction for training split (mode="split")
        val_frac   : fraction for validation split (mode="split")
        seed       : random seed for split (default 42)

    Returns dict with mode-specific keys.
    """
    if mode == "scaffold":
        scaffolds = []
        for smi in smiles:
            try:
                scaffolds.append(murcko_scaffold(smi, generic=generic))
            except ValueError:
                scaffolds.append("")
        return {"scaffolds": scaffolds, "n_molecules": len(smiles)}

    elif mode == "split":
        train, val, test = scaffold_split(
            smiles, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        return {
            "train": train, "val": val, "test": test,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        }

    elif mode == "cluster":
        from collections import defaultdict
        clusters: dict[str, list[str]] = defaultdict(list)
        for smi in smiles:
            try:
                sc = murcko_scaffold(smi, generic=generic) or "__acyclic__"
            except ValueError:
                sc = "__invalid__"
            clusters[sc].append(smi)
        return {
            "clusters": dict(clusters),
            "n_scaffolds": len(clusters),
            "n_molecules": len(smiles),
        }

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'scaffold', 'split', or 'cluster'.")
