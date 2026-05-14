"""
Per-skill eval: fingerprint skill — validates SKILL.md contract.
"""
import json
import pathlib
import torch
import pytest
import molcore

SKILL_DIR = pathlib.Path(__file__).parents[2] / "skills" / "fingerprint"
EXAMPLES   = [json.loads(l) for l in (SKILL_DIR / "examples.jsonl").read_text().splitlines() if l.strip()]


def test_skill_md_exists():
    assert (SKILL_DIR / "SKILL.md").exists()


def test_allowlist_exists():
    assert (SKILL_DIR / "allowlist.yaml").exists()


@pytest.mark.parametrize("ex", [e for e in EXAMPLES if "expected_shape" in e])
def test_golden_shape(ex):
    inp    = ex["input"]
    result = molcore.featurize_smiles(
        inp["smiles"],
        backend = inp.get("backend", "rust"),
        radius  = inp.get("radius", 2),
        nbits   = inp.get("nbits", 2048),
    )
    assert list(result.shape) == ex["expected_shape"], f"shape mismatch: {result.shape}"


@pytest.mark.parametrize("ex", [e for e in EXAMPLES if "expected_error" in e])
def test_golden_invalid_smiles_returns_zeros(ex):
    # Batch API: invalid SMILES → zero fingerprint (graceful degradation),
    # not a raised exception. Exceptions only from Mol.from_smiles() (single-mol).
    result = molcore.featurize_smiles(ex["input"]["smiles"], backend=ex["input"].get("backend", "rust"))
    assert int(result.sum()) == 0, "invalid SMILES in batch should produce all-zero fingerprint"


def test_batch_is_contiguous():
    t = molcore.featurize_smiles(["CCO", "c1ccccc1"], backend="rust")
    assert t.is_contiguous()


def test_dtype_uint8():
    t = molcore.featurize_smiles(["CCO"], backend="rust")
    assert t.dtype == torch.uint8
