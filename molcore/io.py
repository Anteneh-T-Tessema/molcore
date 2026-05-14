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

    # -------------------------------------------------------------------------
    # Slicing and filtering
    # -------------------------------------------------------------------------

    def _subset(self, indices: list[int]) -> "MolDataset":
        """Return a new MolDataset containing only the rows at `indices`."""
        idx = list(indices)
        return MolDataset(
            smiles       = [self.smiles[i] for i in idx],
            fingerprints = self.fingerprints[idx] if self.fingerprints is not None else None,
            descriptors  = self.descriptors[idx]  if self.descriptors  is not None else None,
            labels       = self.labels[idx]        if self.labels       is not None else None,
            metadata     = {k: [v[i] for i in idx] for k, v in self.metadata.items()},
        )

    def filter(self, smarts: str, invert: bool = False) -> "MolDataset":
        """
        Return a new MolDataset containing only molecules that match `smarts`.

        invert=True keeps molecules that do NOT match (e.g. remove reactive groups).
        Invalid SMILES are silently excluded from both branches.
        """
        from molcore.rdkit_bridge import filter_by_smarts
        hits = set(filter_by_smarts(self.smiles, smarts, invert=invert))
        indices = [i for i, smi in enumerate(self.smiles) if smi in hits]
        return self._subset(indices)

    def scaffold_split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> tuple["MolDataset", "MolDataset", "MolDataset"]:
        """
        Scaffold-aware split into train / val / test datasets.

        Returns three MolDataset instances that share no Murcko scaffold.
        Preserves fingerprints, descriptors, labels, and metadata in each split.
        """
        from molcore.rdkit_bridge import scaffold_split as _split
        train_smi, val_smi, test_smi = _split(
            self.smiles, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        smi_to_idx = {smi: i for i, smi in enumerate(self.smiles)}

        def _ds(smi_list: list[str]) -> "MolDataset":
            idx = [smi_to_idx[s] for s in smi_list if s in smi_to_idx]
            return self._subset(idx)

        return _ds(train_smi), _ds(val_smi), _ds(test_smi)

    # -------------------------------------------------------------------------
    # PyG / DGL export
    # -------------------------------------------------------------------------

    def to_pyg_list(self) -> list:
        """
        Convert each molecule to a `torch_geometric.data.Data` object.

        Labels (if present) are attached as `data.y` (float32 scalar or vector).
        Returns a list suitable for use with `torch_geometric.loader.DataLoader`.
        """
        from molcore.molecule import Mol
        graphs = []
        for i, smi in enumerate(self.smiles):
            try:
                data = Mol.from_smiles(smi).to_pyg()
                if self.labels is not None:
                    import torch
                    lbl = self.labels[i]
                    data.y = torch.tensor(
                        [float(lbl)] if np.ndim(lbl) == 0 else lbl.tolist(),
                        dtype=torch.float32,
                    )
                graphs.append(data)
            except Exception:
                pass
        return graphs

    # -------------------------------------------------------------------------
    # 3D descriptor enrichment
    # -------------------------------------------------------------------------

    def add_descriptors_3d(self, seed: int = 42, skip_errors: bool = True) -> "MolDataset":
        """
        Compute 3D shape descriptors (PMI, asphericity, etc.) for every molecule
        and store them in `metadata` under keys like 'pmi1', 'asphericity', etc.

        Returns a new MolDataset (original is not mutated).
        skip_errors=True silently stores NaN for molecules that fail embedding.
        """
        from molcore.rdkit_bridge import calc_descriptors_3d
        import math

        _NAN_DESC = {k: float("nan") for k in
                     ("pmi1", "pmi2", "pmi3", "asphericity", "eccentricity",
                      "npr1", "npr2", "radius_of_gyration",
                      "inertial_shape_factor", "spherocity_index")}

        all_desc: list[dict] = []
        for smi in self.smiles:
            try:
                all_desc.append(calc_descriptors_3d(smi, seed=seed))
            except Exception:
                if skip_errors:
                    all_desc.append(_NAN_DESC.copy())
                else:
                    raise

        new_meta = dict(self.metadata)
        for key in _NAN_DESC:
            new_meta[key] = [d[key] for d in all_desc]

        return MolDataset(
            smiles       = self.smiles,
            fingerprints = self.fingerprints,
            descriptors  = self.descriptors,
            labels       = self.labels,
            metadata     = new_meta,
        )

    # -------------------------------------------------------------------------
    # Database factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_chembl(
        cls,
        query: str,
        limit: int = 100,
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
    ) -> "MolDataset":
        """
        Build a MolDataset from a ChEMBL full-text search.

        Fetches up to `limit` compounds matching `query`, keeps only those with
        valid SMILES, and pre-computes fingerprints and descriptors.
        Attaches ChEMBL IDs and compound names in `metadata`.
        """
        from molcore.databases import chembl_search
        compounds = chembl_search(query, limit=limit)
        valid = [(c.smiles, c.chembl_id, c.name) for c in compounds if c.smiles]
        if not valid:
            return cls(smiles=[], metadata={})
        smiles, ids, names = zip(*valid)
        ds = cls.from_smiles(
            list(smiles),
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
        )
        ds.metadata["chembl_id"] = list(ids)
        ds.metadata["name"] = list(names)
        return ds

    @classmethod
    def from_zinc(
        cls,
        tranche: str = "Drug-Like",
        max_mols: int = 500,
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
    ) -> "MolDataset":
        """
        Build a MolDataset from a ZINC20 tranche download.

        Downloads up to `max_mols` SMILES from the named ZINC tranche,
        then pre-computes fingerprints and descriptors.
        """
        from molcore.databases import zinc_download_tranche
        smiles = zinc_download_tranche(tranche, max_mols=max_mols)
        if not smiles:
            return cls(smiles=[], metadata={})
        return cls.from_smiles(
            smiles,
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __repr__(self) -> str:
        has_fps  = self.fingerprints is not None
        has_desc = self.descriptors  is not None
        has_3d   = "asphericity" in self.metadata
        return (
            f"MolDataset(n={len(self)}, "
            f"fps={'yes' if has_fps else 'no'}, "
            f"desc={'yes' if has_desc else 'no'}, "
            f"3d={'yes' if has_3d else 'no'})"
        )
