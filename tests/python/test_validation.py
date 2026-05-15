"""
Direct unit tests for molcore._validation — all run without RDKit or extras.
"""
import pathlib
import pytest

from molcore._validation import validate_smiles, validate_molblock, validate_path


# ---------------------------------------------------------------------------
# validate_smiles
# ---------------------------------------------------------------------------

def test_validate_smiles_passes_valid():
    assert validate_smiles("CCO") == "CCO"


def test_validate_smiles_returns_unchanged():
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    assert validate_smiles(smi) is smi


def test_validate_smiles_rejects_non_string():
    with pytest.raises(TypeError, match="str"):
        validate_smiles(42)


def test_validate_smiles_rejects_none():
    with pytest.raises(TypeError, match="str"):
        validate_smiles(None)


def test_validate_smiles_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds"):
        validate_smiles("C" * 10_001)


def test_validate_smiles_accepts_limit_exactly():
    # 10 000 chars is the boundary — must not raise
    assert len(validate_smiles("C" * 10_000)) == 10_000


def test_validate_smiles_empty_string_ok():
    # Empty string passes validation (RDKit will reject it, not the validator)
    assert validate_smiles("") == ""


# ---------------------------------------------------------------------------
# validate_molblock
# ---------------------------------------------------------------------------

_MINIMAL_MOLBLOCK = """\

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
"""


def test_validate_molblock_passes_valid():
    result = validate_molblock(_MINIMAL_MOLBLOCK)
    assert result is _MINIMAL_MOLBLOCK


def test_validate_molblock_rejects_non_string():
    with pytest.raises(TypeError, match="str"):
        validate_molblock(b"bytes block")


def test_validate_molblock_rejects_oversized():
    big = "X" * (1_024_001)
    with pytest.raises(ValueError, match="exceeds"):
        validate_molblock(big)


def test_validate_molblock_accepts_limit_exactly():
    ok = "X" * 1_024_000
    assert len(validate_molblock(ok)) == 1_024_000


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------

def test_validate_path_existing_file(tmp_path):
    f = tmp_path / "mol.sdf"
    f.write_text("content")
    result = validate_path(f)
    assert result == f.resolve()


def test_validate_path_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_path(tmp_path / "ghost.sdf")


def test_validate_path_write_mode_nonexistent_ok(tmp_path):
    p = tmp_path / "new_file.sdf"
    result = validate_path(p, write=True)
    assert result == p.resolve()


def test_validate_path_rejects_null_byte(tmp_path):
    with pytest.raises(ValueError, match="null byte"):
        validate_path("/tmp/file\x00evil.sdf")


def test_validate_path_allowed_suffix_ok(tmp_path):
    f = tmp_path / "data.sdf"
    f.write_text("")
    validate_path(f, allowed_suffixes=(".sdf", ".sdf.gz"))


def test_validate_path_disallowed_suffix_raises(tmp_path):
    f = tmp_path / "data.exe"
    f.write_text("")
    with pytest.raises(ValueError, match="extension not in allowed"):
        validate_path(f, allowed_suffixes=(".sdf", ".sdf.gz"))


def test_validate_path_compound_suffix_gz(tmp_path):
    f = tmp_path / "data.sdf.gz"
    f.write_bytes(b"fake gz")
    validate_path(f, allowed_suffixes=(".sdf", ".sdf.gz"))


def test_validate_path_accepts_pathlib(tmp_path):
    f = tmp_path / "mol.sdf"
    f.write_text("")
    result = validate_path(pathlib.Path(f))
    assert isinstance(result, pathlib.Path)
