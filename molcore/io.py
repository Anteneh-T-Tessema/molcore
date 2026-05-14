"""
molcore.io — Parquet/Arrow I/O for molecular datasets.

Columnar storage: SMILES strings + pre-computed fingerprints + descriptors.
Uses pyarrow (available transitively through torch) — no extra install needed.

Usage:
    from molcore.io import write_parquet, read_parquet, MolDataset
    ds = MolDataset.from_smiles(smiles_list, compute_fps=True)
    ds.write_parquet("molecules.parquet")
    ds2 = MolDataset.read_parquet("molecules.parquet")
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from molcore.pipeline import featurize_smiles
from molcore.featurizers.descriptors import calc_descriptors


@dataclass
class MolDataset:
    """
    In-memory columnar molecular dataset.

    Columns:
        smiles      : list[str]             — canonical SMILES
        fingerprints: Optional[np.ndarray]  — (N, nbits) uint8
        descriptors : Optional[np.ndarray]  — (N, 3) float32: [mw, logp, heavy_atoms]
        labels      : Optional[np.ndarray]  — (N,) or (N, k) float32, user-supplied
        metadata    : dict[str, list]       — arbitrary per-row string/numeric columns
    """
    smiles:       list[str]
    fingerprints: Optional[np.ndarray] = field(default=None, repr=False)
    descriptors:  Optional[np.ndarray] = field(default=None, repr=False)
    labels:       Optional[np.ndarray] = field(default=None, repr=False)
    metadata:     dict = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_smiles(
        cls,
        smiles: list[str],
        compute_fps:  bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
        nbits: int = 2048,
        radius: int = 2,
    ) -> "MolDataset":
        fps  = featurize_smiles(smiles, backend=fp_backend, radius=radius, nbits=nbits).numpy() if compute_fps  else None
        desc = calc_descriptors(smiles, backend=fp_backend).numpy() if compute_desc else None
        return cls(smiles=smiles, fingerprints=fps, descriptors=desc)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_arrow_table(self) -> pa.Table:
        columns: dict[str, pa.Array] = {
            "smiles": pa.array(self.smiles, type=pa.string()),
        }
        if self.fingerprints is not None:
            # Store fp as fixed-size binary: nbits bytes, 1 bit per byte (compatible with uint8)
            columns["fingerprints"] = pa.array(
                [row.tobytes() for row in self.fingerprints],
                type=pa.binary(self.fingerprints.shape[1]),
            )
            columns["fp_nbits"] = pa.array(
                [self.fingerprints.shape[1]] * len(self.smiles), type=pa.int32()
            )
        if self.descriptors is not None:
            for i, col_name in enumerate(["mw", "logp", "heavy_atoms"]):
                columns[col_name] = pa.array(self.descriptors[:, i].tolist(), type=pa.float32())
        if self.labels is not None:
            if self.labels.ndim == 1:
                columns["label"] = pa.array(self.labels.tolist(), type=pa.float32())
            else:
                for k in range(self.labels.shape[1]):
                    columns[f"label_{k}"] = pa.array(self.labels[:, k].tolist(), type=pa.float32())
        for name, vals in self.metadata.items():
            columns[name] = pa.array(vals)
        return pa.table(columns)

    def write_parquet(self, path: str | pathlib.Path, compression: str = "snappy") -> None:
        pq.write_table(self.to_arrow_table(), str(path), compression=compression)

    @classmethod
    def read_parquet(cls, path: str | pathlib.Path) -> "MolDataset":
        table  = pq.read_table(str(path))
        smiles = table.column("smiles").to_pylist()

        fps = None
        if "fingerprints" in table.schema.names:
            nbits = table.column("fp_nbits")[0].as_py()
            fps = np.stack([
                np.frombuffer(row.as_py(), dtype=np.uint8)
                for row in table.column("fingerprints")
            ])

        desc = None
        if "mw" in table.schema.names:
            desc = np.column_stack([
                table.column("mw").to_pylist(),
                table.column("logp").to_pylist(),
                table.column("heavy_atoms").to_pylist(),
            ]).astype(np.float32)

        labels = None
        if "label" in table.schema.names:
            labels = np.array(table.column("label").to_pylist(), dtype=np.float32)

        meta_cols = {
            n for n in table.schema.names
            if n not in {"smiles", "fingerprints", "fp_nbits", "mw", "logp", "heavy_atoms", "label"}
        }
        metadata = {n: table.column(n).to_pylist() for n in meta_cols}

        return cls(smiles=smiles, fingerprints=fps, descriptors=desc, labels=labels, metadata=metadata)

    # -------------------------------------------------------------------------
    # Tensor conversion
    # -------------------------------------------------------------------------

    def fingerprints_tensor(self) -> torch.Tensor:
        if self.fingerprints is None:
            raise ValueError("No fingerprints computed — call from_smiles(compute_fps=True)")
        return torch.from_numpy(self.fingerprints)

    def descriptors_tensor(self) -> torch.Tensor:
        if self.descriptors is None:
            raise ValueError("No descriptors computed — call from_smiles(compute_desc=True)")
        return torch.from_numpy(self.descriptors)

    def labels_tensor(self) -> torch.Tensor:
        if self.labels is None:
            raise ValueError("No labels — set ds.labels = np.array(...)")
        return torch.from_numpy(self.labels)

    def __len__(self) -> int:
        return len(self.smiles)

    def __repr__(self) -> str:
        has_fps  = self.fingerprints is not None
        has_desc = self.descriptors  is not None
        return (
            f"MolDataset(n={len(self)}, "
            f"fps={'yes' if has_fps else 'no'}, "
            f"desc={'yes' if has_desc else 'no'})"
        )
