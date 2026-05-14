"""
molcore.predictor — PropertyPredictor: GCN-based molecular property prediction.

Wraps the full train → validate → predict loop so callers never touch PyTorch
training boilerplate directly.

Quick start:
    from molcore.predictor import PropertyPredictor
    from molcore.io import MolDataset
    import numpy as np

    ds = MolDataset.from_smiles(smiles, compute_fps=True)
    ds.labels = np.array(logp_values, dtype=np.float32)

    train_ds, val_ds, _ = ds.scaffold_split()

    pred = PropertyPredictor(hidden=64, epochs=100)
    pred.fit(train_ds, val_dataset=val_ds)

    predictions = pred.predict(["CCO", "c1ccccc1"])   # numpy array
    pred.save("logp_model.pt")

    pred2 = PropertyPredictor.load("logp_model.pt")
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GCN model
# ---------------------------------------------------------------------------

class _MolGCN(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 9,
        hidden: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self._pool = global_mean_pool
        self.convs = torch.nn.ModuleList()
        dims = [in_features] + [hidden] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(d_in, d_out))
        self.dropout = torch.nn.Dropout(dropout)
        self.lin = torch.nn.Linear(hidden, n_outputs)

    def forward(self, data) -> torch.Tensor:
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = self.dropout(x)
        x = self._pool(x, batch)
        return self.lin(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class PropertyPredictor:
    """
    GCN-based molecular property predictor.

    Attributes:
        hidden    : hidden dimension per GCN layer (default 64)
        n_layers  : number of GCN message-passing layers (default 3)
        dropout   : dropout rate (default 0.1)
        epochs    : training epochs (default 150)
        lr        : initial learning rate (default 5e-3)
        batch_size: molecules per batch (default 32)
        device    : 'cpu', 'cuda', or 'auto' (default 'auto')
        n_outputs : number of output values per molecule (default 1)
    """
    hidden:     int   = 64
    n_layers:   int   = 3
    dropout:    float = 0.1
    epochs:     int   = 150
    lr:         float = 5e-3
    batch_size: int   = 32
    device:     str   = "auto"
    n_outputs:  int   = 1

    _model:    Optional[_MolGCN] = field(default=None, init=False, repr=False)
    _history:  dict              = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset,
        val_dataset=None,
        verbose: bool = True,
    ) -> "PropertyPredictor":
        """
        Train on a MolDataset (must have `.labels`).

        Args:
            dataset     : MolDataset with labels set
            val_dataset : optional MolDataset for validation loss tracking
            verbose     : print loss every 25 epochs (default True)
        Returns self for chaining.
        """
        from torch_geometric.loader import DataLoader

        dev = self._resolve_device()
        graphs = _dataset_to_pyg(dataset)
        if not graphs:
            raise ValueError("No valid molecules in dataset")

        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = (
            DataLoader(_dataset_to_pyg(val_dataset), batch_size=self.batch_size)
            if val_dataset is not None else None
        )

        self._model = _MolGCN(
            in_features=9,
            hidden=self.hidden,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_outputs=self.n_outputs,
        ).to(dev)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=15, min_lr=1e-5
        )

        train_losses, val_losses = [], []
        best_val, best_state = float("inf"), None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(self._model, loader, opt, dev)
            train_losses.append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._eval_epoch(self._model, val_loader, dev)
                val_losses.append(val_loss)
                sched.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                sched.step(train_loss)

            if verbose and epoch % 25 == 0:
                msg = f"  epoch {epoch:>4}  train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                print(msg)

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._history = {"train": train_losses, "val": val_losses}
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, smiles: list[str]) -> np.ndarray:
        """
        Predict property values for a list of SMILES.

        Returns (N,) float32 numpy array (or (N, n_outputs) if n_outputs > 1).
        Molecules that fail to parse are assigned NaN.
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first")

        from molcore.molecule import Mol
        from torch_geometric.loader import DataLoader

        dev = self._resolve_device()
        self._model.to(dev).eval()

        graphs, valid_idx = [], []
        for i, smi in enumerate(smiles):
            try:
                graphs.append(Mol.from_smiles(smi).to_pyg())
                valid_idx.append(i)
            except Exception:
                pass

        if not graphs:
            return np.full(len(smiles), float("nan"), dtype=np.float32)

        loader = DataLoader(graphs, batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(dev)
                preds.append(self._model(batch).cpu())
        preds_t = torch.cat(preds).numpy()

        result = np.full(
            (len(smiles),) if self.n_outputs == 1 else (len(smiles), self.n_outputs),
            float("nan"),
            dtype=np.float32,
        )
        result[valid_idx] = preds_t
        return result

    @property
    def history(self) -> dict:
        """Training history: {'train': [...], 'val': [...]}"""
        return self._history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save model weights + hyperparameters to a .pt file."""
        if self._model is None:
            raise RuntimeError("No trained model to save")
        torch.save({
            "state_dict": self._model.state_dict(),
            "hparams": {
                "hidden": self.hidden, "n_layers": self.n_layers,
                "dropout": self.dropout, "n_outputs": self.n_outputs,
            },
            "history": self._history,
        }, str(path))

    @classmethod
    def load(cls, path: str | pathlib.Path, **kwargs) -> "PropertyPredictor":
        """Load a saved PropertyPredictor from a .pt file."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        hp = ckpt["hparams"]
        pred = cls(
            hidden=hp["hidden"], n_layers=hp["n_layers"],
            dropout=hp["dropout"], n_outputs=hp["n_outputs"],
            **kwargs,
        )
        pred._model = _MolGCN(
            in_features=9,
            hidden=hp["hidden"],
            n_layers=hp["n_layers"],
            dropout=hp["dropout"],
            n_outputs=hp["n_outputs"],
        )
        pred._model.load_state_dict(ckpt["state_dict"])
        pred._model.eval()
        pred._history = ckpt.get("history", {})
        return pred

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def score(self, dataset) -> dict[str, float]:
        """
        Evaluate on a labelled MolDataset. Returns R², MAE, RMSE.
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first")
        graphs = _dataset_to_pyg(dataset)
        smiles = [dataset.smiles[i] for i in range(len(dataset))]
        preds = self.predict(smiles)
        targets = dataset.labels.flatten()
        mask = ~np.isnan(preds)
        p, t = preds[mask], targets[mask]
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae  = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        return {"r2": r2, "mae": mae, "rmse": rmse, "n": int(mask.sum())}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            from molcore.gpu import best_device
            return best_device()
        return torch.device(self.device)

    @staticmethod
    def _train_epoch(model, loader, optimizer, device) -> float:
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
        return total / count if count else float("nan")

    @staticmethod
    @torch.no_grad()
    def _eval_epoch(model, loader, device) -> float:
        model.eval()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            loss = F.mse_loss(model(batch), batch.y)
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
        return total / count if count else float("nan")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dataset_to_pyg(dataset) -> list:
    """Convert MolDataset with labels to a list of PyG Data objects."""
    from molcore.molecule import Mol
    graphs = []
    for i, smi in enumerate(dataset.smiles):
        try:
            data = Mol.from_smiles(smi).to_pyg()
            if dataset.labels is not None:
                lbl = dataset.labels[i]
                data.y = torch.tensor(
                    [float(lbl)] if np.ndim(lbl) == 0 else lbl.tolist(),
                    dtype=torch.float32,
                )
            graphs.append(data)
        except Exception:
            pass
    return graphs
