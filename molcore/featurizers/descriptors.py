import numpy as np
import torch
from molcore._molcore import calc_descriptors_batch as _calc_descriptors_batch


def calc_descriptors(smiles: list[str], backend: str = "rust") -> torch.Tensor:
    """
    Batch molecular descriptors: MW, LogP, heavy atom count.

    backend="rust"  → fast Rust approximation (MW exact, LogP Crippen-simplified)
    backend="rdkit" → exact RDKit values (MW, Crippen LogP, TPSA)
    """
    if backend == "rust":
        np_arr = _calc_descriptors_batch(smiles)
        return torch.from_numpy(np_arr)
    elif backend == "rdkit":
        from molcore.rdkit_bridge import calc_descriptors_rdkit
        return calc_descriptors_rdkit(smiles)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def calc_named_descriptors(
    smiles: list[str],
    names: "list[str] | None" = None,
    preset: "str | None" = None,
) -> "tuple[np.ndarray, list[str]]":
    """
    Compute any named RDKit 2D descriptors for a batch of SMILES.

    names: explicit list of descriptor names, e.g. ["MolWt", "TPSA", "NumHDonors"]
    preset: "lipinski" | "druglike" | "all"
    Returns (array (N, D) float32, list_of_descriptor_names).
    Invalid SMILES rows are filled with NaN.
    """
    from molcore.rdkit_bridge import calc_named_descriptors as _calc
    return _calc(smiles, names=names, preset=preset)


def list_descriptor_names() -> list[str]:
    """Return all ~200 available RDKit 2D descriptor names."""
    from molcore.rdkit_bridge import list_descriptor_names as _list
    return _list()


DESCRIPTOR_PRESETS: "dict[str, list[str]]" = {
    "lipinski": [
        "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "RingCount",
    ],
    "druglike": [
        "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "RingCount",
        "FractionCSP3", "HeavyAtomCount", "NumAromaticRings",
        "NumAliphaticRings", "NumSaturatedRings", "BertzCT",
        "MolMR", "LabuteASA",
    ],
}
