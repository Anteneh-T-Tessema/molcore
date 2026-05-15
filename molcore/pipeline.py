"""
pipeline.py — batch-first primary entry point.
Always call this; per-mol methods are convenience wrappers around it.
"""
import torch
from observability.metrics.skill_metrics import timed as _timed


def featurize_smiles(
    smiles: list[str],
    kind: str = "ecfp4",
    backend: str = "rust",
    radius: int = 2,
    nbits: int = 2048,
) -> torch.Tensor:
    """
    Batch featurization.

    backend="rust"  → Rust Rayon parallel, zero-copy numpy → torch
                      Use for: training new models, billion-scale screening
    backend="rdkit" → RDKit via rdkit_bridge, bit-identical to legacy models
                      Use for: inference on models trained on RDKit fingerprints

    Callers decide the backend — it is never auto-selected.
    """
    with _timed("fingerprint", batch_size=len(smiles)):
        kind_lc = kind.lower()

        if kind_lc == "ecfp4":
            if backend == "rust":
                from molcore.featurizers.fingerprints import ecfp4
                return ecfp4(smiles, radius=radius, nbits=nbits)
            elif backend == "rdkit":
                from molcore.rdkit_bridge import ecfp4_rdkit
                return ecfp4_rdkit(smiles, radius=radius, nbits=nbits)
            else:
                raise ValueError(f"Unknown backend: {backend!r}. Choose 'rust' or 'rdkit'.")

        # Non-ECFP4 types always use the RDKit bridge (Rust doesn't implement them yet)
        if kind_lc == "maccs":
            from molcore.rdkit_bridge import maccs_keys
            return torch.from_numpy(maccs_keys(smiles))
        elif kind_lc in ("atom_pairs", "atompairs"):
            from molcore.rdkit_bridge import atom_pairs_fp
            return torch.from_numpy(atom_pairs_fp(smiles, nbits=nbits))
        elif kind_lc in ("topological_torsions", "torsions"):
            from molcore.rdkit_bridge import topological_torsions_fp
            return torch.from_numpy(topological_torsions_fp(smiles, nbits=nbits))
        elif kind_lc == "rdkit":
            from molcore.rdkit_bridge import rdkit_path_fp
            return torch.from_numpy(rdkit_path_fp(smiles, nbits=nbits))
        else:
            raise ValueError(
                f"Unsupported fingerprint kind: {kind!r}. "
                "Choose: 'ecfp4', 'maccs', 'atom_pairs', 'topological_torsions', 'rdkit'."
            )
