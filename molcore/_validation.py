"""Input sanitization for public molcore APIs."""
from __future__ import annotations

import pathlib

# Pathological SMILES (e.g. a million-atom polymer) can exhaust CPU/memory in
# RDKit's ring-perception code. This limit is well above any real drug molecule.
_MAX_SMILES_LEN = 10_000

# MDL Mol blocks are usually <10 KB; 1 MB is a generous upper bound.
_MAX_MOLBLOCK_LEN = 1_024_000


def validate_smiles(smiles: str) -> str:
    """
    Reject non-string or excessively long SMILES before handing off to RDKit.

    Raises TypeError or ValueError on bad input, returns the original string
    unchanged on success so callers can use it inline::

        smi = validate_smiles(smi)
    """
    if not isinstance(smiles, str):
        raise TypeError(f"SMILES must be str, got {type(smiles).__name__!r}")
    if len(smiles) > _MAX_SMILES_LEN:
        raise ValueError(
            f"SMILES length {len(smiles):,} exceeds limit of {_MAX_SMILES_LEN:,} "
            "characters — possible resource-exhaustion input"
        )
    return smiles


def validate_molblock(block: str) -> str:
    """
    Reject non-string or oversized MDL Mol blocks before handing off to RDKit.
    """
    if not isinstance(block, str):
        raise TypeError(f"molblock must be str, got {type(block).__name__!r}")
    if len(block) > _MAX_MOLBLOCK_LEN:
        raise ValueError(
            f"Mol block size {len(block):,} bytes exceeds limit of "
            f"{_MAX_MOLBLOCK_LEN // 1024:,} KB — possible resource-exhaustion input"
        )
    return block


def validate_path(
    path: "str | pathlib.Path",
    *,
    write: bool = False,
    allowed_suffixes: "tuple[str, ...] | None" = None,
) -> pathlib.Path:
    """
    Resolve and validate a user-supplied file path.

    Checks:
    - No null bytes (can truncate paths on some systems and bypass extension checks).
    - Extension is in ``allowed_suffixes`` if provided (handles compound suffixes
      like ``.sdf.gz``).
    - File exists when ``write=False``.

    Returns the resolved ``pathlib.Path`` on success.
    """
    raw = str(path)
    if "\x00" in raw:
        raise ValueError("File path contains a null byte")

    p = pathlib.Path(path).resolve()

    if allowed_suffixes is not None:
        name = p.name.lower()
        if not any(name.endswith(s) for s in allowed_suffixes):
            raise ValueError(
                f"File {p.name!r}: extension not in allowed set {allowed_suffixes}"
            )

    if not write and not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    return p
