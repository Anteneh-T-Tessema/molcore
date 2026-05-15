"""
molcore.io — Parquet/Arrow I/O for molecular datasets.

Columnar storage: SMILES strings + pre-computed fingerprints + descriptors.
Uses pyarrow (declared as a core dependency in pyproject.toml).

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
from molcore._validation import validate_path, validate_smiles


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
    def from_sdf(
        cls,
        path: "str | pathlib.Path",
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
        nbits: int = 2048,
        radius: int = 2,
        sanitize: bool = True,
        remove_hs: bool = True,
    ) -> "MolDataset":
        """
        Load molecules from an SDF or gzipped SDF (.sdf.gz) file.

        All SD properties are stored in `metadata`. Invalid records are silently skipped.
        """
        from molcore.rdkit_bridge import from_sdf_file, mol_to_smiles, canonicalize
        validate_path(path, allowed_suffixes=(".sdf", ".gz", ".sdf.gz"))
        records = from_sdf_file(str(path), sanitize=sanitize, remove_hs=remove_hs)
        if not records:
            return cls(smiles=[], metadata={})
        smiles_list, all_props = [], []
        for rdmol, props in records:
            try:
                smi = canonicalize(mol_to_smiles(rdmol))
                smiles_list.append(smi)
                all_props.append(props)
            except Exception:
                pass
        # collect property keys across all records
        prop_keys: set[str] = set()
        for p in all_props:
            prop_keys.update(p.keys())
        metadata = {k: [p.get(k, "") for p in all_props] for k in sorted(prop_keys)}
        ds = cls.from_smiles(
            smiles_list,
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
            nbits=nbits,
            radius=radius,
        )
        ds.metadata.update(metadata)
        return ds

    def write_sdf(
        self,
        path: "str | pathlib.Path",
        extra_props: "dict[str, list] | None" = None,
    ) -> None:
        """
        Write this dataset to an SDF file.

        Metadata columns and `extra_props` are written as SD properties.
        """
        from molcore.rdkit_bridge import write_sdf as _write_sdf
        validate_path(path, write=True, allowed_suffixes=(".sdf", ".gz", ".sdf.gz"))
        props = dict(self.metadata)
        if extra_props:
            props.update(extra_props)
        if self.labels is not None:
            if self.labels.ndim == 1:
                props["label"] = self.labels.tolist()
            else:
                for k in range(self.labels.shape[1]):
                    props[f"label_{k}"] = self.labels[:, k].tolist()
        _write_sdf(self.smiles, str(path), properties=props or None)

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
        p = validate_path(path, write=True, allowed_suffixes=(".parquet",))
        pq.write_table(self.to_arrow_table(), str(p), compression=compression)

    @classmethod
    def read_parquet(cls, path: str | pathlib.Path) -> "MolDataset":
        p = validate_path(path, allowed_suffixes=(".parquet",))
        table  = pq.read_table(str(p))
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
        schema_names = table.schema.names
        _reserved = {"smiles", "fingerprints", "fp_nbits", "mw", "logp", "heavy_atoms", "label"}
        if "label" in schema_names:
            labels = np.array(table.column("label").to_pylist(), dtype=np.float32)
        else:
            # multi-label: label_0, label_1, ...
            label_cols = sorted(
                [n for n in schema_names if n.startswith("label_") and n[6:].isdigit()],
                key=lambda n: int(n[6:]),
            )
            if label_cols:
                labels = np.column_stack(
                    [table.column(c).to_pylist() for c in label_cols]
                ).astype(np.float32)
                _reserved = _reserved | set(label_cols)

        meta_cols = {n for n in schema_names if n not in _reserved}
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
        from collections import defaultdict
        train_smi, val_smi, test_smi = _split(
            self.smiles, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        # Build a list of positions per SMILES to handle duplicate SMILES correctly.
        # A plain dict overwrites earlier indices; a deque lets us pop in order.
        from collections import deque as _deque
        smi_to_positions: dict[str, _deque] = defaultdict(_deque)
        for i, smi in enumerate(self.smiles):
            smi_to_positions[smi].append(i)

        def _ds(smi_list: list[str]) -> "MolDataset":
            idx = []
            for s in smi_list:
                if smi_to_positions[s]:
                    idx.append(smi_to_positions[s].popleft())
            return self._subset(idx)

        train_ds = _ds(train_smi)
        val_ds   = _ds(val_smi)
        test_ds  = _ds(test_smi)
        return train_ds, val_ds, test_ds

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
                    if np.ndim(lbl) == 0:
                        data.y = torch.tensor([float(lbl)], dtype=torch.float32)
                    else:
                        data.y = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
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

    # -------------------------------------------------------------------------
    # Pandas bridge
    # -------------------------------------------------------------------------

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert to a pandas DataFrame.

        Columns: 'smiles', optionally 'fp_{i}' (if fingerprints present),
        descriptor columns ('mw','logp','heavy_atoms'), label columns,
        and all metadata columns. Requires pandas.
        """
        import pandas as _pd
        data: dict = {"smiles": self.smiles}
        if self.descriptors is not None:
            for i, col in enumerate(["mw", "logp", "heavy_atoms"]):
                data[col] = self.descriptors[:, i].tolist()
        if self.labels is not None:
            if self.labels.ndim == 1:
                data["label"] = self.labels.tolist()
            else:
                for k in range(self.labels.shape[1]):
                    data[f"label_{k}"] = self.labels[:, k].tolist()
        data.update(self.metadata)
        return _pd.DataFrame(data)

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        smiles_col: str = "smiles",
        label_col: "str | None" = None,
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
    ) -> "MolDataset":
        """
        Build a MolDataset from a pandas DataFrame.

        smiles_col: name of the SMILES column.
        label_col:  optional column to use as labels (float32).
        All other non-smiles, non-label columns become metadata.
        """
        smiles = df[smiles_col].tolist()
        ds = cls.from_smiles(
            smiles,
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
        )
        if label_col and label_col in df.columns:
            ds.labels = df[label_col].to_numpy(dtype=np.float32)
        skip = {smiles_col, label_col}
        for col in df.columns:
            if col not in skip:
                ds.metadata[col] = df[col].tolist()
        return ds

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def draw_grid(
        self,
        n: int = 20,
        mols_per_row: int = 4,
        sub_img_size: "tuple[int, int]" = (200, 150),
        legend_col: "str | None" = None,
    ) -> str:
        """
        Render the first `n` molecules as a grid SVG.

        legend_col: metadata column to use as per-molecule legend (default: SMILES).
        Returns an SVG string — displays inline in Jupyter.
        """
        from molcore.rdkit_bridge import mols_to_grid_svg
        subset = self.smiles[:n]
        if legend_col and legend_col in self.metadata:
            legends = [str(v) for v in self.metadata[legend_col][:n]]
        else:
            legends = None
        return mols_to_grid_svg(subset, mols_per_row=mols_per_row,
                                 sub_img_size=sub_img_size, legends=legends)

    # -------------------------------------------------------------------------
    # Diversity picking
    # -------------------------------------------------------------------------

    def diversity_pick(
        self,
        n: int,
        nbits: int = 2048,
        radius: int = 2,
        seed: int = 0,
    ) -> "MolDataset":
        """
        Select `n` maximally diverse molecules using the MaxMin algorithm.

        Iteratively picks the molecule with the highest minimum Tanimoto distance
        to all already-selected molecules. Returns a new MolDataset of exactly `n`
        molecules (fewer if the dataset has fewer valid molecules).

        Typical use: pare down a large enumerated library to a diverse screening set.

        Args:
            n     : number of molecules to select
            nbits : fingerprint length for diversity calculation (default 2048)
            radius: Morgan fingerprint radius (default 2)
            seed  : starting molecule index; varied across runs for ensemble picking
        """
        from molcore.rdkit_bridge import diversity_pick as _pick
        indices = _pick(self.smiles, n=n, nbits=nbits, radius=radius, seed=seed)
        return self._subset(indices)

    # -------------------------------------------------------------------------
    # Cross-validation splits
    # -------------------------------------------------------------------------

    def kfold(
        self,
        k: int = 5,
        seed: int = 42,
    ) -> "list[tuple[MolDataset, MolDataset]]":
        """
        Random k-fold cross-validation split.

        Returns k (train, val) MolDataset pairs. Each molecule appears in
        exactly one validation fold and k-1 training folds.
        """
        import random as _random
        n = len(self)
        if k < 2:
            raise ValueError(f"k must be ≥ 2, got {k}")
        if n < k:
            raise ValueError(f"Dataset has {n} molecules but k={k}; need at least k molecules")
        idx = list(range(n))
        _random.Random(seed).shuffle(idx)
        folds = [idx[i::k] for i in range(k)]
        result = []
        for fold_i in range(k):
            val_idx = folds[fold_i]
            train_idx = [i for j, fold in enumerate(folds) for i in fold if j != fold_i]
            result.append((self._subset(train_idx), self._subset(val_idx)))
        return result

    def scaffold_kfold(
        self,
        k: int = 5,
        seed: int = 42,
    ) -> "list[tuple[MolDataset, MolDataset]]":
        """
        Scaffold-aware k-fold cross-validation.

        Whole Murcko scaffold groups are assigned to folds, so no scaffold
        appears in both train and val within any fold.
        Returns k (train, val) MolDataset pairs.
        """
        from molcore.rdkit_bridge import murcko_scaffold
        from collections import defaultdict
        import random as _random

        if k < 2:
            raise ValueError(f"k must be ≥ 2, got {k}")

        scaffold_groups: dict[str, list[int]] = defaultdict(list)
        for i, smi in enumerate(self.smiles):
            try:
                sc = murcko_scaffold(smi)
            except Exception:
                sc = smi
            scaffold_groups[sc].append(i)

        groups = list(scaffold_groups.values())
        _random.Random(seed).shuffle(groups)

        folds: list[list[int]] = [[] for _ in range(k)]
        for g_i, group in enumerate(groups):
            folds[g_i % k].extend(group)

        result = []
        for fold_i in range(k):
            val_idx = folds[fold_i]
            train_idx = [i for j, fold in enumerate(folds) for i in fold if j != fold_i]
            result.append((self._subset(train_idx), self._subset(val_idx)))
        return result

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    def cluster(
        self,
        cutoff: float = 0.4,
        nbits: int = 2048,
        radius: int = 2,
    ) -> "MolDataset":
        """
        Cluster molecules using the Butina algorithm and store cluster IDs in metadata.

        cutoff: Tanimoto distance threshold (1 - similarity). Default 0.4
                → molecules with similarity ≥ 0.6 end up in the same cluster.

        Returns a new MolDataset (original unchanged) with a 'cluster_id'
        metadata column added. Cluster 0 is the largest cluster.
        """
        from molcore.rdkit_bridge import butina_cluster
        ids = butina_cluster(self.smiles, cutoff=cutoff, nbits=nbits, radius=radius)
        new_meta = dict(self.metadata)
        new_meta["cluster_id"] = ids
        return MolDataset(
            smiles       = self.smiles,
            fingerprints = self.fingerprints,
            descriptors  = self.descriptors,
            labels       = self.labels,
            metadata     = new_meta,
        )

    # -------------------------------------------------------------------------
    # TDC / BindingDB factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_tdc(
        cls,
        dataset: str,
        split: str = "train",
        split_method: str = "scaffold",
        log_transform: bool = True,
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
    ) -> "MolDataset":
        """
        Build a MolDataset from any TDC ADMET or DTI benchmark dataset.

        Parameters
        ----------
        dataset : str
            TDC dataset name, e.g. ``"BBB_Martini"``, ``"hERG"``,
            ``"BindingDB_Kd"``, ``"Davis"``.
        split : str
            Which split to load: ``"train"``, ``"valid"``, or ``"test"``.
        split_method : str
            TDC split strategy: ``"scaffold"`` (default), ``"random"``,
            or ``"cold_drug"`` (DTI only).
        log_transform : bool
            For regression endpoints in nM units (Kd, IC50, Ki), convert
            labels to ``pIC50 = -log10(Y * 1e-9)`` if True.
        compute_fps : bool
            Pre-compute Morgan ECFP4 fingerprints.
        compute_desc : bool
            Pre-compute physicochemical descriptors.

        Requires ``pip install molcore[bio]`` (PyTDC).

        Example::

            ds = MolDataset.from_tdc("BBB_Martini", split="train")
            # ds.smiles   → SMILES strings
            # ds.labels   → binary BBB labels (0/1)
        """
        from molcore.databases import tdc_dataset
        import math

        splits = tdc_dataset(dataset, split_method=split_method)
        df = splits[split]

        smiles_list = df["Drug"].tolist()
        y_raw = df["Y"].to_numpy(dtype=np.float32)

        # Detect regression endpoints by fractional labels
        is_regression = not set(np.unique(y_raw)).issubset({0.0, 1.0})

        if is_regression and log_transform:
            # nM → pIC50 conversion; clamp to avoid log(0)
            y = np.array(
                [-math.log10(max(v * 1e-9, 1e-15)) for v in y_raw],
                dtype=np.float32,
            )
        else:
            y = y_raw

        ds = cls.from_smiles(
            smiles_list,
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
        )
        ds.labels = y
        ds.metadata["tdc_dataset"] = [dataset] * len(smiles_list)
        ds.metadata["tdc_split"]   = [split]   * len(smiles_list)

        # If DTI dataset: store protein sequences in metadata
        if "Target" in df.columns:
            ds.metadata["protein_sequence"] = df["Target"].tolist()
        if "Target_ID" in df.columns:
            ds.metadata["target_id"] = df["Target_ID"].tolist()

        return ds

    @classmethod
    def from_bindingdb(
        cls,
        affinity: str = "Kd",
        target: str | None = None,
        split: str = "train",
        log_transform: bool = True,
        max_records: int = 5_000,
        compute_fps: bool = True,
        compute_desc: bool = True,
        fp_backend: str = "rust",
    ) -> "MolDataset":
        """
        Build a MolDataset from BindingDB bioactivity data via TDC.

        Parameters
        ----------
        affinity : str
            Measurement type: ``"Kd"``, ``"IC50"``, ``"Ki"``, or ``"EC50"``.
        target : str or None
            Optional substring filter on protein name/sequence (e.g. ``"EGFR"``).
        split : str
            TDC data split: ``"train"``, ``"valid"``, or ``"test"``.
        log_transform : bool
            Convert nM values to pIC50 = -log10(Y × 1e-9).
        max_records : int
            Cap on number of records loaded.

        The returned dataset has:
        - ``ds.labels``  → pIC50 (or raw nM if log_transform=False)
        - ``ds.metadata["protein_sequence"]`` → target amino acid sequences
        - ``ds.metadata["affinity_type"]``    → affinity measurement type
        - ``ds.metadata["target_id"]``        → target identifiers (if available)

        Requires ``pip install molcore[bio]`` (PyTDC).

        Example::

            ds = MolDataset.from_bindingdb("Kd", target="EGFR")
            # Fine-tune a property predictor on EGFR binding affinity
            pred = PropertyPredictor(hidden=128, epochs=200)
            train, val, test = ds.scaffold_split()
            pred.fit(train, val_dataset=val)
        """
        from molcore.databases import bindingdb_search
        import math

        records = bindingdb_search(
            affinity=affinity, target=target,
            max_records=max_records, split=split,
        )
        if not records:
            return cls(smiles=[], metadata={})

        smiles_list = [r.smiles for r in records]
        y_raw = np.array([r.affinity for r in records], dtype=np.float32)

        if log_transform:
            y = np.array(
                [-math.log10(max(v * 1e-9, 1e-15)) for v in y_raw],
                dtype=np.float32,
            )
        else:
            y = y_raw

        ds = cls.from_smiles(
            smiles_list,
            compute_fps=compute_fps,
            compute_desc=compute_desc,
            fp_backend=fp_backend,
        )
        ds.labels = y
        ds.metadata["protein_sequence"] = [r.target_sequence for r in records]
        ds.metadata["affinity_type"]    = [r.affinity_type    for r in records]
        ds.metadata["target_id"]        = [r.target_name      for r in records]
        return ds

    def _repr_html_(self) -> str:
        """Jupyter HTML: summary + 8-molecule grid."""
        from molcore.rdkit_bridge import mols_to_grid_svg
        grid = mols_to_grid_svg(self.smiles[:8], mols_per_row=4, sub_img_size=(180, 130))
        has_fps  = "yes" if self.fingerprints is not None else "no"
        has_desc = "yes" if self.descriptors  is not None else "no"
        return (
            f"<b>MolDataset</b> n={len(self)} | fps={has_fps} | desc={has_desc}"
            f"<br/>{grid}"
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> "MolDataset":
        """Integer index → single-row MolDataset. Supports negative indexing."""
        if idx < 0:
            idx = len(self) + idx
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        return self._subset([idx])

    def __iter__(self):
        """Iterate over rows, each yielding a single-row MolDataset."""
        for i in range(len(self)):
            yield self._subset([i])

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


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class MolTorchDataset:
    """
    Wraps a MolDataset as a PyTorch Dataset for use with DataLoader.

    Each item is a `torch_geometric.data.Data` object with:
      - `data.x`          : (N, 9) float32 node features
      - `data.edge_index` : (2, E) int64
      - `data.edge_attr`  : (E, 4) float32
      - `data.y`          : float32 label scalar (if dataset.labels is not None)
      - `data.smiles`     : str canonical SMILES

    Usage:
        from torch_geometric.loader import DataLoader
        torch_ds = MolTorchDataset(ds)
        loader = DataLoader(torch_ds, batch_size=32, shuffle=True)
    """

    def __init__(self, dataset: MolDataset):
        self._ds = dataset
        self._graphs: Optional[list] = None  # lazy-built

    def _build(self) -> None:
        from molcore.molecule import Mol
        import torch
        graphs = []
        for i, smi in enumerate(self._ds.smiles):
            try:
                data = Mol.from_smiles(smi).to_pyg()
                data.smiles = smi
                if self._ds.labels is not None:
                    lbl = self._ds.labels[i]
                    if np.ndim(lbl) == 0:
                        data.y = torch.tensor([float(lbl)], dtype=torch.float32)
                    else:
                        data.y = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
                graphs.append(data)
            except Exception:
                graphs.append(None)
        self._graphs = graphs

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        if self._graphs is None:
            self._build()
        item = self._graphs[idx]
        if item is None:
            raise ValueError(f"Molecule at index {idx} ({self._ds.smiles[idx]!r}) could not be parsed")
        return item

    def __repr__(self) -> str:
        return f"MolTorchDataset(n={len(self)}, built={self._graphs is not None})"
