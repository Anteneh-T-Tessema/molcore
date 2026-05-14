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
