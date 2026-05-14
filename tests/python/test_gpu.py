"""Tests for molcore.gpu — run on CPU (no CUDA required)."""
import torch
import pytest

from molcore.gpu import (
    best_device,
    device_info,
    to_device,
    to_gpu,
    to_cpu,
    tanimoto_gpu,
    autocast_context,
    nvmolkit_available,
)


def test_best_device_returns_device():
    dev = best_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda", "mps")


def test_device_info_has_cpu():
    info = device_info()
    assert info["cpu"] is True
    assert "cuda" in info
    assert "mps"  in info


def test_to_device_cpu_noop():
    t = torch.ones(4)
    out = to_device(t, "cpu")
    assert out.device.type == "cpu"
    # same storage when already on target
    assert out.data_ptr() == t.data_ptr()


def test_to_gpu_single_tensor():
    t   = torch.ones(4)
    out = to_gpu(t, device="cpu")
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"


def test_to_gpu_list_of_tensors():
    ts  = [torch.ones(3), torch.zeros(5)]
    out = to_gpu(ts, device="cpu")
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, torch.Tensor) for x in out)


def test_to_cpu_returns_cpu_tensor():
    t = torch.ones(4)
    assert to_cpu(t).device.type == "cpu"


def test_tanimoto_gpu_shape():
    q = torch.randint(0, 2, (3, 2048), dtype=torch.float32)
    l = torch.randint(0, 2, (5, 2048), dtype=torch.float32)
    sim = tanimoto_gpu(q, l, device="cpu")
    assert sim.shape == (3, 5)


def test_tanimoto_gpu_self_similarity_one():
    fps = torch.randint(0, 2, (4, 2048), dtype=torch.float32)
    sim = tanimoto_gpu(fps, fps, device="cpu")
    diag = sim.diagonal()
    assert (diag - 1.0).abs().max() < 1e-5, "self-similarity must be 1.0"


def test_tanimoto_gpu_range():
    fps = torch.randint(0, 2, (4, 128), dtype=torch.float32)
    sim = tanimoto_gpu(fps, fps, device="cpu")
    assert sim.min() >= 0.0
    assert sim.max() <= 1.0 + 1e-6


def test_tanimoto_gpu_chunk_matches_full():
    torch.manual_seed(0)
    q = torch.randint(0, 2, (4, 256), dtype=torch.float32)
    l = torch.randint(0, 2, (8, 256), dtype=torch.float32)
    full   = tanimoto_gpu(q, l, device="cpu", chunk_size=8)
    chunked = tanimoto_gpu(q, l, device="cpu", chunk_size=3)
    assert (full - chunked).abs().max() < 1e-5


def test_autocast_context_cpu():
    ctx = autocast_context("cpu")
    with ctx:
        t = torch.ones(4) + torch.ones(4)
    assert t.dtype == torch.float32


def test_nvmolkit_available_returns_bool():
    result = nvmolkit_available()
    assert isinstance(result, bool)
