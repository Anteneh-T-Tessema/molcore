"""
Tool: conformer — 3D conformer generation and shape descriptor calculation.
Strict function signature for agent tool-use.
"""
from __future__ import annotations
from molcore.rdkit_bridge import generate_conformers, calc_descriptors_3d


def run(
    smiles: str | list[str],
    mode: str = "descriptors_3d",
    n_confs: int = 1,
    seed: int = 42,
    force_field: str = "MMFF94",
) -> dict:
    """
    Args:
        smiles      : single SMILES or list (mode="batch_descriptors")
        mode        : "descriptors_3d"   → shape descriptor dict for one molecule
                      "coordinates"      → (n_atoms, 3) float64 coordinates
                      "batch_descriptors"→ list of descriptor dicts for smiles list
        n_confs     : number of conformers to generate (mode="coordinates")
        seed        : ETKDGv3 random seed (default 42)
        force_field : "MMFF94" | "MMFF94s" | "UFF"

    Returns dict with mode-specific keys.
    """
    if mode == "descriptors_3d":
        if isinstance(smiles, list):
            smiles = smiles[0]
        try:
            desc = calc_descriptors_3d(smiles, seed=seed)
            return {"smiles": smiles, "descriptors": desc, "seed": seed}
        except ValueError as e:
            return {"smiles": smiles, "error": str(e)}

    elif mode == "coordinates":
        if isinstance(smiles, list):
            smiles = smiles[0]
        try:
            confs = generate_conformers(smiles, n_confs=n_confs, seed=seed, force_field=force_field)
            return {
                "smiles": smiles,
                "n_conformers": len(confs),
                "n_atoms": confs[0].shape[0] if confs else 0,
                "coordinates": [c.tolist() for c in confs],
            }
        except ValueError as e:
            return {"smiles": smiles, "error": str(e)}

    elif mode == "batch_descriptors":
        if isinstance(smiles, str):
            smiles = [smiles]
        results = []
        for smi in smiles:
            try:
                desc = calc_descriptors_3d(smi, seed=seed)
                results.append({"smiles": smi, "descriptors": desc})
            except ValueError as e:
                results.append({"smiles": smi, "error": str(e)})
        return {"results": results, "n_molecules": len(smiles)}

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'descriptors_3d', 'coordinates', or 'batch_descriptors'.")
