"""
End-to-end example: train a 2-layer GCN on molecular property prediction.

Demonstrates the full molcore pipeline:
  SMILES → Mol (Rust ingest) → PyG Data (zero-copy) → GCN → predict logP

Uses a small hand-curated dataset so no external download is required.
Run with:
    python examples/end_to_end_gnn.py
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from molcore.molecule import Mol

# Small curated dataset: (SMILES, RDKit-LogP)
DATASET = [
    ("CCO",                      -0.14),
    ("CCCO",                     -0.53),  # 1-propanol
    ("CCCCO",                     0.80),  # 1-butanol
    ("c1ccccc1",                  1.90),  # benzene
    ("Cc1ccccc1",                 2.27),  # toluene
    ("c1ccc(cc1)C",               2.27),  # also toluene (different SMILES)
    ("CC(=O)O",                  -0.17),  # acetic acid
    ("CCC(=O)O",                  0.33),  # propanoic acid
    ("c1ccccc1C(=O)O",            1.87),  # benzoic acid
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 3.72), # ibuprofen-like
    ("c1ccc2ccccc2c1",            3.30),  # naphthalene
    ("CCCCc1ccccc1",              3.98),  # butylbenzene
    ("c1ccc(nc1)N",              -0.67),  # 2-aminopyridine
    ("CC(=O)Nc1ccc(O)cc1",       -0.02),  # paracetamol
    ("OC(=O)c1ccccc1O",           1.19),  # salicylic acid
    ("c1ccncc1",                  0.65),  # pyridine
    ("CCCC",                      2.05),  # butane
    ("CC(C)C",                    1.96),  # isobutane
    ("CCN(CC)CC",                 1.45),  # triethylamine
    ("c1ccc(cc1)O",               1.46),  # phenol
]


# ---------------------------------------------------------------------------
# 2-layer GCN
# ---------------------------------------------------------------------------

class MolGCN(torch.nn.Module):
    def __init__(self, node_features: int = 4, hidden: int = 32):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_pyg_dataset(records):
    graphs = []
    for smiles, logp in records:
        try:
            mol  = Mol.from_smiles(smiles)
            data = mol.to_pyg()
            data.y = torch.tensor([logp], dtype=torch.float32)
            graphs.append(data)
        except Exception as e:
            print(f"  skip {smiles!r}: {e}")
    return graphs


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------

def train(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / sum(b.num_graphs for b in loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        preds.append(model(batch))
        targets.append(batch.y)
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    ss_res  = ((preds - targets) ** 2).sum()
    ss_tot  = ((targets - targets.mean()) ** 2).sum()
    r2      = 1 - ss_res / ss_tot
    mae     = (preds - targets).abs().mean()
    return float(r2), float(mae)


def main():
    print("molcore end-to-end GCN example")
    print("=" * 50)

    print(f"\nBuilding PyG dataset from {len(DATASET)} molecules...")
    graphs = build_pyg_dataset(DATASET)
    print(f"  {len(graphs)} graphs built (zero-copy Rust → numpy → torch)")

    # 80/20 train/test split
    split    = int(0.8 * len(graphs))
    train_ds = graphs[:split]
    test_ds  = graphs[split:]

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8)

    model     = MolGCN(node_features=4, hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"\nTraining 2-layer GCN for 100 epochs...")
    print(f"  train: {len(train_ds)} mols  test: {len(test_ds)} mols")
    print()

    for epoch in range(1, 101):
        loss = train(model, train_loader, optimizer)
        if epoch % 20 == 0:
            r2, mae = evaluate(model, test_loader)
            print(f"  epoch {epoch:3d} | train loss {loss:.4f} | test R² {r2:.3f} | test MAE {mae:.3f}")

    r2, mae = evaluate(model, test_loader)
    print(f"\nFinal: R² = {r2:.3f}, MAE = {mae:.3f} log units")

    if r2 > 0.5:
        print("PASS — R² > 0.5 on held-out set (reasonable for 16 training mols)")
    else:
        print("NOTE — R² below 0.5; expected with only 16 training molecules")

    # Show sample predictions vs ground truth
    print("\nSample predictions:")
    print(f"  {'SMILES':<45} {'Pred':>6} {'True':>6}")
    model.eval()
    with torch.no_grad():
        for data, (smiles, logp) in zip(test_ds, DATASET[split:]):
            batch = next(iter(DataLoader([data], batch_size=1)))
            pred  = model(batch)
            print(f"  {smiles:<45} {pred.item():>6.2f} {logp:>6.2f}")


if __name__ == "__main__":
    main()
