"""
molcore.gpu — CUDA tensor utilities for molecular workloads.

Provides device-aware transfer helpers that work gracefully when CUDA is
absent (CPU fallback) and thin wrappers for nvMolKit ops when available.

Design rule: every public function accepts and returns torch.Tensor so the
caller never needs to branch on device availability.
"""
from __future__ import annotations

from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def best_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a summary dict of available compute devices."""
    info: dict = {"cpu": True, "cuda": False, "mps": False, "nvmolkit": False}
    if torch.cuda.is_available():
        info["cuda"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"]  = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"]    = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps"] = True
    try:
        import nvmolkit  # type: ignore
        info["nvmolkit"] = True
        info["nvmolkit_version"] = getattr(nvmolkit, "__version__", "unknown")
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Tensor transfer helpers
# ---------------------------------------------------------------------------

def to_device(
    tensor: torch.Tensor,
    device: str | torch.device | None = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    """
    Move a tensor to `device` (default: best_device()).
    Returns the same tensor if it is already on the target device.
    """
    dev = torch.device(device) if device is not None else best_device()
    if tensor.device == dev:
        return tensor
    return tensor.to(dev, non_blocking=non_blocking)


def to_gpu(
    tensors: torch.Tensor | Sequence[torch.Tensor],
    device: str | torch.device | None = None,
    non_blocking: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    """Move one tensor or a list of tensors to GPU (best_device() by default)."""
    dev = torch.device(device) if device is not None else best_device()
    if isinstance(tensors, torch.Tensor):
        return to_device(tensors, dev, non_blocking)
    return [to_device(t, dev, non_blocking) for t in tensors]


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU. No-op if already on CPU."""
    return tensor.cpu()


# ---------------------------------------------------------------------------
# Batched similarity on GPU
# ---------------------------------------------------------------------------

def tanimoto_gpu(
    query_fps: torch.Tensor,
    library_fps: torch.Tensor,
    device: str | torch.device | None = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Tanimoto similarity matrix computed entirely on GPU.

    query_fps   : (Q, B) uint8 or float32
    library_fps : (L, B) uint8 or float32
    Returns     : (Q, L) float32 on CPU

    Falls back to CPU computation when CUDA/MPS is unavailable.
    Chunked to avoid OOM on large libraries.
    """
    dev = torch.device(device) if device is not None else best_device()

    q = query_fps.float().to(dev)
    l = library_fps.float().to(dev)

    Q, L = q.shape[0], l.shape[0]
    result = torch.empty(Q, L, dtype=torch.float32)

    for start in range(0, L, chunk_size):
        end   = min(start + chunk_size, L)
        l_chunk = l[start:end]                        # (chunk, B)
        inter   = torch.mm(q, l_chunk.T)              # (Q, chunk) — AND popcount approx
        q_bits  = q.sum(dim=1, keepdim=True)          # (Q, 1)
        l_bits  = l_chunk.sum(dim=1, keepdim=True).T  # (1, chunk)
        union   = q_bits + l_bits - inter
        sim     = inter / union.clamp(min=1e-9)
        result[:, start:end] = sim.cpu()

    return result


# ---------------------------------------------------------------------------
# nvMolKit bridge (optional)
# ---------------------------------------------------------------------------

def nvmolkit_available() -> bool:
    """Return True if nvMolKit is installed and importable."""
    try:
        import nvmolkit  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def nvmolkit_fingerprints(
    smiles: list[str],
    fp_type: str = "ECFP4",
    nbits: int = 2048,
) -> torch.Tensor:
    """
    Compute fingerprints using NVIDIA nvMolKit (GPU-accelerated).

    Falls back to molcore Rust pipeline if nvMolKit is not available.
    fp_type: 'ECFP4' | 'ECFP6' | 'FCFP4' — passed directly to nvMolKit.
    Returns: (N, nbits) uint8 tensor on CPU.

    Requires: pip install nvmolkit  (NVIDIA Volta+ GPU, CUDA 11.2+)
    """
    try:
        import nvmolkit  # type: ignore
        fps = nvmolkit.fingerprints(smiles, fp_type=fp_type, n_bits=nbits)
        if isinstance(fps, torch.Tensor):
            return fps.cpu().byte()
        return torch.tensor(fps, dtype=torch.uint8)
    except ImportError:
        # Graceful fallback to Rust pipeline
        from molcore.pipeline import featurize_smiles
        return featurize_smiles(smiles, backend="rust", nbits=nbits)


# ---------------------------------------------------------------------------
# Mixed-precision helpers
# ---------------------------------------------------------------------------

def autocast_context(device: str | torch.device | None = None):
    """
    Return a torch.autocast context manager for the given device.
    Returns a no-op context on CPU (autocast only benefits CUDA/MPS).
    """
    dev = torch.device(device) if device is not None else best_device()
    if dev.type in ("cuda", "mps"):
        return torch.autocast(device_type=dev.type, dtype=torch.float16)
    # CPU: return a harmless nullcontext
    from contextlib import nullcontext
    return nullcontext()
