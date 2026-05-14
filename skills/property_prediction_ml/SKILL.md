# Skill: property_prediction_ml

## When to invoke
When an agent needs a trained model to predict continuous molecular properties (LogP, solubility,
IC50, pKa) from SMILES, using graph-based representation learning instead of hand-crafted
descriptors.

## Architecture

**PropertyPredictor** wraps a 3-layer GCN (Graph Convolutional Network) over the molecular graph.
Node features: 9-dimensional (`atomic_num`, `is_aromatic`, `formal_charge`, `num_hs`, `degree`,
`in_ring`, `hybridization`, `chirality`, `mass_norm`). Global mean pooling → linear head.

## Quick start

```python
from molcore.predictor import PropertyPredictor
from molcore.io import MolDataset
import numpy as np

smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
logp   = np.array([-0.14, 1.68, -0.17, -0.13], dtype=np.float32)

ds = MolDataset.from_smiles(smiles, compute_fps=False, compute_desc=False)
ds.labels = logp

train_ds, val_ds, test_ds = ds.scaffold_split()

pred = PropertyPredictor(hidden=64, epochs=100, lr=5e-3)
pred.fit(train_ds, val_dataset=val_ds, verbose=True)

preds = pred.predict(["CC", "CCCC"])   # → (2,) float32 ndarray
metrics = pred.score(test_ds)          # → {r2, mae, rmse, n}
```

## Hyperparameter reference

| Parameter | Default | Description |
|---|---|---|
| `hidden` | `64` | GCN hidden dimension |
| `n_layers` | `3` | Message-passing layers |
| `dropout` | `0.1` | Dropout rate (training only) |
| `epochs` | `150` | Training epochs |
| `lr` | `5e-3` | Initial Adam learning rate |
| `batch_size` | `32` | Molecules per gradient step |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` / `"mps"` |
| `n_outputs` | `1` | Outputs per molecule (1 = regression) |

Scheduler: `ReduceLROnPlateau(factor=0.5, patience=15)` on validation loss.  
Best-val checkpoint is restored automatically at end of training.

## Uncertainty estimation (MC Dropout)

```python
mean, std = pred.predict_with_uncertainty(smiles, n_samples=30)
# mean: (N,) float32 — point estimate
# std:  (N,) float32 — epistemic uncertainty proxy
```

## Persistence

```python
pred.save("logp_gcn.pt")
pred2 = PropertyPredictor.load("logp_gcn.pt")
```

Checkpoint stores: `state_dict`, `hparams` (hidden/n_layers/dropout/n_outputs), `history`.

## Multi-task prediction

```python
pred = PropertyPredictor(n_outputs=3, hidden=128, epochs=200)
# ds.labels must be (N, 3) float32
```

## Training history

```python
import matplotlib.pyplot as plt
plt.plot(pred.history["train"], label="train")
plt.plot(pred.history["val"],   label="val")
```

## PyTorch DataLoader integration

```python
from molcore.io import MolTorchDataset
from torch_geometric.loader import DataLoader

torch_ds = MolTorchDataset(ds)
loader = DataLoader(torch_ds, batch_size=32, shuffle=True)
```

## Evaluation metrics

`pred.score(dataset)` returns:
- `r2`   — coefficient of determination
- `mae`  — mean absolute error
- `rmse` — root mean squared error
- `n`    — number of valid molecules scored

## When NOT to use
- Do not use without a labelled training set — this is supervised learning, not zero-shot.
- Default 150 epochs may overfit on very small datasets (<50 molecules); reduce `epochs` or increase `dropout`.
- For classification tasks, subclass `PropertyPredictor` and swap the MSE loss for BCE.
