"""
molcore.dti — Drug-Target Interaction predictor.

Architecture
------------
Ligand encoder  : GCN/GAT/GIN on PyG molecular graph  →  hidden-dim embedding
Protein encoder : 1D CNN on one-hot residue features   →  hidden-dim embedding
Joint head      : MLP([lig_emb ‖ prot_emb])            →  scalar affinity (pIC50)

No extra dependencies required — the protein encoder uses one-hot residue
features that are always available.  ESM-2 embeddings are supported when
``pip install molcore-chem[bio]`` is installed.

Quick start::

    from molcore.dti import DTIPredictor, DTIDataset

    ds = DTIDataset(
        smiles    = ["CC(=O)O",      "c1ccccc1"],
        sequences = ["MKTLLILAVL",   "ACDEFGHIKL"],
        labels    = [6.5,             7.2],        # pIC50
    )
    pred = DTIPredictor(hidden=64, epochs=50)
    pred.fit(ds)

    pred.save("dti_model.pt")
    pred2 = DTIPredictor.load("dti_model.pt")
    affinities = pred2.predict(["CCO"], ["MKTLLILAVL"])

Load data from BindingDB (requires ``pip install molcore-chem[bio]``)::

    from molcore.databases import bindingdb_search
    records = bindingdb_search(affinity="Kd", max_records=1000)
    ds = DTIDataset.from_bindingdb_records(records)
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard amino acids, alphabetical
_AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_VOCAB)}
_AA_DIM = len(_AA_VOCAB)             # 20


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class _LigandEncoder(torch.nn.Module):
    """GNN that returns a per-molecule embedding (no prediction head)."""

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        model_type: str = "gcn",
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
        self._pool      = global_mean_pool
        self.dropout    = torch.nn.Dropout(dropout)
        self.model_type = model_type
        self.convs      = torch.nn.ModuleList()

        dims = [9] + [hidden] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            if model_type == "gcn":
                self.convs.append(GCNConv(d_in, d_out))
            elif model_type == "gat":
                heads = 4
                out   = d_out // heads if d_out % heads == 0 else d_out
                self.convs.append(GATConv(d_in, out, heads=heads, concat=True,
                                          dropout=dropout))
            elif model_type == "gin":
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(d_in, d_out), torch.nn.ReLU(),
                    torch.nn.Linear(d_out, d_out),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            else:
                raise ValueError(f"Unknown model_type {model_type!r}. "
                                 "Choose: 'gcn', 'gat', 'gin'")

        if model_type == "gat":
            heads = 4
            self.out_dim = (hidden // heads if hidden % heads == 0 else hidden) * heads
        else:
            self.out_dim = hidden

    def forward(self, data) -> torch.Tensor:
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, ei)) if self.model_type == "gat" else F.relu(conv(x, ei))
            x = self.dropout(x)
        return self._pool(x, batch)   # (B, out_dim)


class _ProteinCNN(torch.nn.Module):
    """
    Lightweight 1D CNN protein encoder — no extra dependencies.

    Input : (B, 20, L) one-hot residue tensor
    Output: (B, hidden) global representation
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1   = torch.nn.Conv1d(_AA_DIM, hidden, kernel_size=3, padding=1)
        self.conv2   = torch.nn.Conv1d(hidden,  hidden, kernel_size=3, padding=1)
        self.pool    = torch.nn.AdaptiveMaxPool1d(1)
        self.drop    = torch.nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))    # (B, hidden, L)
        x = self.drop(x)
        x = F.relu(self.conv2(x))    # (B, hidden, L)
        return self.pool(x).squeeze(-1)  # (B, hidden)


class _DTINet(torch.nn.Module):
    """Joint ligand-protein network → scalar binding affinity."""

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        model_type: str = "gcn",
    ):
        super().__init__()
        self.lig_enc  = _LigandEncoder(hidden, n_layers, dropout, model_type)
        self.prot_enc = _ProteinCNN(hidden, dropout)
        joint_dim     = self.lig_enc.out_dim + self.prot_enc.out_dim
        self.head = torch.nn.Sequential(
            torch.nn.Linear(joint_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, mol_batch, prot: torch.Tensor) -> torch.Tensor:
        lig  = self.lig_enc(mol_batch)             # (B, lig_dim)
        prot = self.prot_enc(prot)                 # (B, prot_dim)
        return self.head(torch.cat([lig, prot], dim=-1)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DTIDataset:
    """
    Paired (SMILES, protein sequence, affinity) dataset.

    Attributes:
        smiles    : list of ligand SMILES strings
        sequences : list of protein amino acid sequences (single-letter codes)
        labels    : optional list of binding affinities (pIC50 or similar)
    """
    smiles:    list[str]
    sequences: list[str]
    labels:    "list[float] | None" = None

    def __post_init__(self) -> None:
        if len(self.smiles) != len(self.sequences):
            raise ValueError("smiles and sequences must have the same length")
        if self.labels is not None and len(self.labels) != len(self.smiles):
            raise ValueError("labels length must match smiles/sequences length")

    def __len__(self) -> int:
        return len(self.smiles)

    @classmethod
    def from_bindingdb_records(cls, records: list) -> "DTIDataset":
        """
        Build from a list of :class:`~molcore.databases.BindingRecord` objects.

        Records with ``affinity=nan`` are included but will be skipped during
        training. Typical usage::

            from molcore.databases import bindingdb_search
            records = bindingdb_search(affinity="Kd", max_records=1000)
            ds = DTIDataset.from_bindingdb_records(records)
        """
        return cls(
            smiles    = [r.smiles           for r in records],
            sequences = [r.target_sequence  for r in records],
            labels    = [r.affinity         for r in records],
        )

    def scaffold_split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> "tuple[DTIDataset, DTIDataset, DTIDataset]":
        """
        Scaffold-aware split so no Murcko scaffold leaks between splits.
        Falls back to random split if rdkit_bridge is unavailable.
        """
        from molcore.rdkit_bridge import scaffold_split as _spl
        train_smi, val_smi, test_smi = _spl(
            self.smiles, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        smi_set = {s: i for i, s in enumerate(self.smiles)}

        def _subset(smi_list: list[str]) -> "DTIDataset":
            idx = [smi_set[s] for s in smi_list if s in smi_set]
            return DTIDataset(
                smiles    = [self.smiles[i]    for i in idx],
                sequences = [self.sequences[i] for i in idx],
                labels    = [self.labels[i]    for i in idx] if self.labels else None,
            )

        return _subset(train_smi), _subset(val_smi), _subset(test_smi)


# ---------------------------------------------------------------------------
# Helper: sequence → one-hot tensor
# ---------------------------------------------------------------------------

def _seq_to_onehot(seq: str, max_len: int = 1000) -> torch.Tensor:
    """Return (20, min(L, max_len)) one-hot tensor for an amino acid sequence."""
    seq = seq[:max_len].upper()
    x   = torch.zeros(_AA_DIM, len(seq))
    for j, aa in enumerate(seq):
        idx = _AA_TO_IDX.get(aa)
        if idx is not None:
            x[idx, j] = 1.0
    return x


def _pad_proteins(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack variable-length (20, L) tensors into (B, 20, max_L) with zero padding."""
    max_L  = max(t.shape[1] for t in tensors)
    out    = torch.zeros(len(tensors), _AA_DIM, max_L)
    for i, t in enumerate(tensors):
        out[i, :, :t.shape[1]] = t
    return out


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

@dataclass
class DTIPredictor:
    """
    Drug-Target Interaction predictor — same fit/predict/save/load API
    as :class:`~molcore.predictor.PropertyPredictor`.

    Attributes:
        hidden      : hidden dimension for both encoders (default 64)
        n_layers    : GNN message-passing layers (default 3)
        dropout     : dropout rate (default 0.1)
        epochs      : training epochs (default 100)
        lr          : learning rate (default 1e-3)
        batch_size  : pairs per batch (default 32)
        max_seq_len : protein sequence truncation length (default 1000)
        model_type  : ligand GNN — 'gcn' (default), 'gat', or 'gin'
        device      : 'cpu', 'cuda', or 'auto' (default 'auto')
    """
    hidden:      int   = 64
    n_layers:    int   = 3
    dropout:     float = 0.1
    epochs:      int   = 100
    lr:          float = 1e-3
    batch_size:  int   = 32
    max_seq_len: int   = 1000
    model_type:  str   = "gcn"
    device:      str   = "auto"

    _model:   Optional[_DTINet] = field(default=None, init=False, repr=False)
    _history: dict              = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: DTIDataset,
        val_dataset: "DTIDataset | None" = None,
        verbose: bool = True,
    ) -> "DTIPredictor":
        """
        Train on a labelled DTIDataset.

        Args:
            dataset     : training data with labels
            val_dataset : optional validation set for LR scheduling + best-model tracking
            verbose     : print loss every 25 epochs (default True)
        Returns self for chaining.
        """
        if dataset.labels is None:
            raise ValueError("DTIDataset must have labels to call fit()")

        dev = self._resolve_device()
        self._model = _DTINet(self.hidden, self.n_layers, self.dropout, self.model_type).to(dev)
        opt   = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=10, min_lr=1e-5
        )

        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val, best_state = float("inf"), None

        for epoch in range(1, self.epochs + 1):
            tloss = self._run_epoch(dataset, dev, opt, train=True)
            train_losses.append(tloss)

            vloss = None
            if val_dataset is not None:
                vloss = self._run_epoch(val_dataset, dev, opt=None, train=False)
                val_losses.append(vloss)
                sched.step(vloss)
                if vloss < best_val:
                    best_val  = vloss
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                sched.step(tloss)

            if verbose and epoch % 25 == 0:
                msg = f"  epoch {epoch:>4}  train={tloss:.4f}"
                if vloss is not None:
                    msg += f"  val={vloss:.4f}"
                print(msg)

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._history = {"train": train_losses, "val": val_losses}
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, smiles: list[str], sequences: list[str]) -> np.ndarray:
        """
        Predict binding affinities for (smiles, sequence) pairs.

        Returns (N,) float32 array. Pairs that fail to parse receive NaN.
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first")
        if len(smiles) != len(sequences):
            raise ValueError("smiles and sequences must have the same length")

        model = self._model
        dev = self._resolve_device()
        model.to(dev).eval()

        results = np.full(len(smiles), float("nan"), dtype=np.float32)
        for start in range(0, len(smiles), self.batch_size):
            smi_b = smiles[start:start + self.batch_size]
            seq_b = sequences[start:start + self.batch_size]
            graphs, prot_ts, valid_local = self._parse_batch(smi_b, seq_b)
            if not graphs:
                continue
            from torch_geometric.data import Batch
            mb = Batch.from_data_list(graphs).to(dev)  # type: ignore[union-attr]
            pt = _pad_proteins(prot_ts).to(dev)
            with torch.no_grad():
                preds = model(mb, pt).cpu().numpy()
            for pred_i, local_i in enumerate(valid_local):
                results[start + local_i] = preds[pred_i]

        return results

    def score(self, dataset: DTIDataset) -> dict[str, float]:
        """Compute R², MAE, RMSE on a labelled DTIDataset."""
        if dataset.labels is None:
            raise ValueError("DTIDataset must have labels")
        preds  = self.predict(dataset.smiles, dataset.sequences)
        labels = np.array(dataset.labels, dtype=np.float32)
        mask   = ~np.isnan(preds) & ~np.isnan(labels)
        p, t   = preds[mask], labels[mask]
        if len(p) == 0:
            return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "n": 0}
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        return {
            "r2":   float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "mae":  float(np.abs(p - t).mean()),
            "rmse": float(np.sqrt(((p - t) ** 2).mean())),
            "n":    int(mask.sum()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "str | pathlib.Path") -> None:
        """Save model weights and hyperparameters to a .pt checkpoint."""
        if self._model is None:
            raise RuntimeError("No trained model to save — call fit() first")
        torch.save({
            "state_dict": self._model.state_dict(),
            "hparams": {
                "hidden":      self.hidden,
                "n_layers":    self.n_layers,
                "dropout":     self.dropout,
                "max_seq_len": self.max_seq_len,
                "model_type":  self.model_type,
            },
            "history": self._history,
        }, str(path))

    @classmethod
    def load(cls, path: "str | pathlib.Path", **kwargs) -> "DTIPredictor":
        """Load a DTIPredictor from a .pt checkpoint."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        hp   = ckpt["hparams"]
        pred = cls(
            hidden      = hp["hidden"],
            n_layers    = hp["n_layers"],
            dropout     = hp["dropout"],
            max_seq_len = hp.get("max_seq_len", 1000),
            model_type  = hp.get("model_type", "gcn"),
            **kwargs,
        )
        pred._model = _DTINet(
            hp["hidden"], hp["n_layers"], hp["dropout"], hp.get("model_type", "gcn")
        )
        pred._model.load_state_dict(ckpt["state_dict"])
        pred._model.eval()
        pred._history = ckpt.get("history", {})
        return pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            from molcore.gpu import best_device
            return best_device()
        return torch.device(self.device)

    def _parse_batch(
        self,
        smi_batch: list[str],
        seq_batch: list[str],
    ) -> "tuple[list, list[torch.Tensor], list[int]]":
        """
        Parse one batch.  Returns (graphs, prot_tensors, valid_local_indices).
        valid_local_indices[i] is the position in smi_batch of the i-th valid pair.
        """
        from molcore.molecule import Mol
        graphs, prot_ts, valid_local = [], [], []
        for local_i, (smi, seq) in enumerate(zip(smi_batch, seq_batch)):
            try:
                graphs.append(Mol.from_smiles(smi).to_pyg())
                prot_ts.append(_seq_to_onehot(seq, self.max_seq_len))
                valid_local.append(local_i)
            except Exception:
                pass
        return graphs, prot_ts, valid_local

    def _run_epoch(
        self,
        dataset: DTIDataset,
        dev: torch.device,
        opt,
        train: bool,
    ) -> float:
        from torch_geometric.data import Batch

        assert self._model is not None
        assert dataset.labels is not None  # fit() checks this before calling _run_epoch
        if train:
            self._model.train()
        else:
            self._model.eval()

        total, count = 0.0, 0
        indices = np.random.permutation(len(dataset)) if train else np.arange(len(dataset))

        for start in range(0, len(indices), self.batch_size):
            idx_b = indices[start:start + self.batch_size]
            smi_b  = [dataset.smiles[i]    for i in idx_b]
            seq_b  = [dataset.sequences[i] for i in idx_b]
            lbl_b  = [dataset.labels[i]    for i in idx_b]

            graphs, prot_ts = [], []
            labels: list[float] = []
            for smi, seq, lbl in zip(smi_b, seq_b, lbl_b):
                if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
                    continue
                try:
                    from molcore.molecule import Mol
                    graphs.append(Mol.from_smiles(smi).to_pyg())
                    prot_ts.append(_seq_to_onehot(seq, self.max_seq_len))
                    labels.append(float(lbl))
                except Exception:
                    pass

            if not graphs:
                continue

            mb = Batch.from_data_list(graphs).to(dev)  # type: ignore[union-attr]
            pt = _pad_proteins(prot_ts).to(dev)
            y  = torch.tensor(labels, dtype=torch.float32, device=dev)

            if train:
                opt.zero_grad()
                loss = F.mse_loss(self._model(mb, pt), y)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    loss = F.mse_loss(self._model(mb, pt), y)

            total += loss.item() * len(labels)
            count += len(labels)

        return total / count if count else float("nan")
