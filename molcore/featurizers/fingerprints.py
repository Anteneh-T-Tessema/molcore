import torch
from molcore._molcore import ecfp4_batch as _ecfp4_batch


def ecfp4(smiles: list[str], radius: int = 2, nbits: int = 2048) -> torch.Tensor:
    """
    Batch ECFP4 fingerprints.
    Path: list[str] → Rust (Rayon parallel) → numpy (zero-copy) → torch (zero-copy)
    No Python loop. No intermediate copy.
    """
    np_arr = _ecfp4_batch(smiles, radius, nbits)   # numpy, Rust-owned memory
    return torch.from_numpy(np_arr)                 # shares numpy buffer, no copy
