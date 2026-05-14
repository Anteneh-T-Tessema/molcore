"""
tools/local/reaction.py — Reaction transform tool.

Modes:
  unimolecular : apply SMARTS to one molecule       → list[str]
  bimolecular  : apply SMARTS to two reactants      → list[str]
  enumerate    : apply SMARTS to a library          → list[str]
"""
from __future__ import annotations
import argparse
import json
import sys


def run(
    smiles: str | list[str],
    rxn_smarts: str,
    mode: str = "unimolecular",
    smiles_b: str | None = None,
    max_products: int = 1000,
) -> list[str]:
    from molcore.rdkit_bridge import react, react_bimolecular, enumerate_reactions

    if mode == "unimolecular":
        if not isinstance(smiles, str):
            raise ValueError("unimolecular mode expects a single SMILES string")
        return react(smiles, rxn_smarts)

    if mode == "bimolecular":
        if not isinstance(smiles, str) or smiles_b is None:
            raise ValueError("bimolecular mode expects smiles (str) and smiles_b (str)")
        return react_bimolecular(smiles, smiles_b, rxn_smarts)

    if mode == "enumerate":
        lib = smiles if isinstance(smiles, list) else [smiles]
        return enumerate_reactions(lib, rxn_smarts, max_products=max_products)

    raise ValueError(f"Unknown mode: {mode!r}. Choose unimolecular | bimolecular | enumerate")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Reaction transform tool")
    p.add_argument("--smiles", required=True, help="SMILES or JSON list of SMILES")
    p.add_argument("--rxn-smarts", required=True)
    p.add_argument("--mode", default="unimolecular",
                   choices=["unimolecular", "bimolecular", "enumerate"])
    p.add_argument("--smiles-b", default=None, help="Second reactant (bimolecular only)")
    p.add_argument("--max-products", type=int, default=1000)
    args = p.parse_args()

    try:
        smiles = json.loads(args.smiles)
    except (json.JSONDecodeError, TypeError):
        smiles = args.smiles

    result = run(
        smiles=smiles,
        rxn_smarts=args.rxn_smarts,
        mode=args.mode,
        smiles_b=args.smiles_b,
        max_products=args.max_products,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
