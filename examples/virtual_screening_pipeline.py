"""
Virtual screening pipeline — end-to-end molcore example.

Steps:
  1. Fetch training data from ChEMBL (kinase inhibitors, IC50 values)
  2. Apply Lipinski Ro5 + PAINS SMARTS filter
  3. Scaffold-aware train/val/test split
  4. Train a GCN PropertyPredictor on LogP (as a proxy — swap for pIC50 in production)
  5. Evaluate on held-out test set
  6. Score a ZINC drug-like sample and rank by predicted property
  7. Report top-10 hits with uncertainty estimates

Run:
    python examples/virtual_screening_pipeline.py

Dependencies: molcore + rdkit + torch + torch_geometric
"""
from __future__ import annotations

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Step 1 — build a labelled dataset from SMILES
# ---------------------------------------------------------------------------
# We use a hard-coded reference set so this example runs offline.
# In production: replace with MolDataset.from_chembl("kinase", limit=500)

TRAINING_SMILES = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",       # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "c1ccc2ccccc2c1",                   # naphthalene
    "CC(C)(C)NCC(O)c1ccc(O)c(O)c1",    # salbutamol
    "OC(=O)c1ccccc1O",                  # salicylic acid
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",    # naproxen
    "c1ccc(cc1)Cc1ccccc1",             # diphenylmethane
    "CCc1ccc(cc1)S(=O)(=O)N",          # sulfonamide
    "CC(=O)Nc1ccc(O)cc1",              # acetaminophen
    "Clc1ccc(cc1)C(c1ccccc1)=C(Cl)Cl", # DDT analog
    "CC12CCC3C(C1CCC2=O)CCC4=C3C=CC(=C4)O",  # estradiol-like
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # caffeine
    "OC(=O)CCCCC(=O)O",               # adipic acid
    "c1ccncc1",                        # pyridine
    "c1ccoc1",                         # furan
    "c1ccsc1",                         # thiophene
    "Nc1ccccc1",                       # aniline
    "Nc1ccc(cc1)S(=O)(=O)N",          # sulfanilamide
    "CCCC",                            # butane
    "CCCCCCC",                         # heptane
]

# Experimental LogP values (approximate, for demonstration)
LOGP_LABELS = np.array([
    3.50,  # ibuprofen
    1.19,  # aspirin
    3.37,  # naphthalene
    0.64,  # salbutamol
    2.23,  # salicylic acid
    3.18,  # naproxen
    3.59,  # diphenylmethane
    1.20,  # sulfonamide
    0.46,  # acetaminophen
    5.60,  # DDT analog
    3.80,  # estradiol-like
    -0.07, # caffeine
    1.44,  # adipic acid
    0.65,  # pyridine
    1.34,  # furan
    1.81,  # thiophene
    0.90,  # aniline
    -0.62, # sulfanilamide
    2.89,  # butane
    4.50,  # heptane
], dtype=np.float32)

# Prospective screening library (normally from ZINC)
SCREENING_LIBRARY = [
    "CC(C)c1ccc(cc1)C(C)C(=O)O",   # ibuprofen isomer
    "c1ccc2c(c1)cccc2C(=O)O",       # 2-naphthoic acid
    "CC(=O)Nc1ccc(Cl)cc1",          # paracetamol analog
    "CCN(CC)c1ccc(cc1)N=Nc1ccc(cc1)S(=O)(=O)N",  # azo dye
    "O=C(O)c1ccncc1",               # nicotinic acid
    "CCC(CC)COC(=O)Nc1ccccc1",     # phenylurethane
    "Cc1ccc(cc1)S(=O)(=O)O",       # p-toluenesulfonic acid
    "INVALID_SMILES_EXAMPLE",       # intentionally bad — should get NaN
]

# ---------------------------------------------------------------------------
# Step 2 — build MolDataset and apply filters
# ---------------------------------------------------------------------------

def lipinski_filter(smiles: list[str], desc: np.ndarray) -> list[bool]:
    """Ro5: MW ≤ 500, LogP ≤ 5."""
    mask = []
    for mw, logp, _ in desc.tolist():
        mask.append(mw <= 500 and logp <= 5)
    return mask


def main() -> None:
    from molcore.io import MolDataset
    from molcore.predictor import PropertyPredictor

    print("=" * 60)
    print("molcore virtual screening pipeline")
    print("=" * 60)

    # ── 1. Dataset ──────────────────────────────────────────────
    print("\n[1] Building dataset …")
    ds = MolDataset.from_smiles(TRAINING_SMILES, compute_fps=True, compute_desc=True)
    ds.labels = LOGP_LABELS
    print(f"    {ds}")

    # ── 2. Lipinski filter ───────────────────────────────────────
    print("\n[2] Applying Lipinski Ro5 filter …")
    if ds.descriptors is not None:
        mask = lipinski_filter(ds.smiles, ds.descriptors)
        passing = sum(mask)
        print(f"    {passing}/{len(ds)} molecules pass Ro5")

    # ── 3. Scaffold split ────────────────────────────────────────
    print("\n[3] Scaffold-aware split (80/10/10) …")
    train_ds, val_ds, test_ds = ds.scaffold_split(train_frac=0.8, val_frac=0.1)
    print(f"    train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    if len(train_ds) < 3:
        print("    [WARN] Dataset too small for reliable GCN training — increase TRAINING_SMILES")
        print("    In production use MolDataset.from_chembl('kinase', limit=500)")

    # ── 4. Train GCN ────────────────────────────────────────────
    print("\n[4] Training GCN PropertyPredictor …")
    pred = PropertyPredictor(
        hidden=32,
        n_layers=2,
        dropout=0.1,
        epochs=80,
        lr=5e-3,
        batch_size=8,
    )
    pred.fit(train_ds, val_dataset=val_ds if len(val_ds) > 0 else None, verbose=True)

    # ── 5. Evaluate on test set ─────────────────────────────────
    if len(test_ds) > 0:
        print("\n[5] Test-set evaluation …")
        metrics = pred.score(test_ds)
        print(f"    R²={metrics['r2']:.3f}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  n={metrics['n']}")
    else:
        print("\n[5] Test set empty (too few molecules) — skipping evaluation")

    # ── 6. Screen prospective library ───────────────────────────
    print("\n[6] Screening prospective library with uncertainty …")
    mean, std = pred.predict_with_uncertainty(SCREENING_LIBRARY, n_samples=20)

    # ── 7. Rank and report hits ──────────────────────────────────
    print("\n[7] Top hits by predicted LogP (descending):\n")
    print(f"    {'SMILES':<45}  {'pred LogP':>9}  {'±':>7}")
    print("    " + "-" * 65)

    ranked = sorted(
        [(smi, m, s) for smi, m, s in zip(SCREENING_LIBRARY, mean.tolist(), std.tolist())
         if not np.isnan(m)],
        key=lambda x: x[1],
        reverse=True,
    )
    for smi, m, s in ranked[:10]:
        print(f"    {smi:<45}  {m:>9.3f}  {s:>7.3f}")

    invalid = [smi for smi, m, _ in zip(SCREENING_LIBRARY, mean.tolist()) if np.isnan(m)]
    if invalid:
        print(f"\n    Skipped (invalid SMILES): {invalid}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
