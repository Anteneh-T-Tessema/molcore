"""
End-to-end benchmark: molcore vs RDKit + manual PyG pipeline.

Measures the full workflow a computational chemist runs before model training:
  1. Parse / load molecules
  2. Standardize (strip salts, neutralize, canonical tautomer)
  3. Compute ECFP4 fingerprints (2048-bit)
  4. Compute 7 Lipinski descriptors
  5. Scaffold-based train/val/test split
  6. Convert to PyG Data objects

Usage:
    python benchmarks/bench_e2e.py --n 1000
    python benchmarks/bench_e2e.py --n 5000 --no-rdkit   # skip RDKit baseline

Results are printed as a summary table and saved to benchmarks/results_e2e.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Synthetic molecule library (repeatable, no external file needed)
# ---------------------------------------------------------------------------

_BASE_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "c1cccnc1", "CC(C)O",
    "CCCO", "CCCCO", "CCc1ccccc1", "Cc1ccccc1", "c1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "O=C(O)c1ccc(N)cc1",
    "CC1=C2C(=O)c3ccccc3C2(CC1=O)O", "Nc1ccc(cc1)S(N)(=O)=O",
    "[Na+].OC(=O)c1ccccc1",  # salt — standardization removes Na
]


def make_smiles(n: int, seed: int = 42) -> list[str]:
    """Generate n SMILES by cycling and shuffling the base set."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(_BASE_SMILES), size=n)
    return [_BASE_SMILES[i] for i in indices]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed(label: str, fn: Callable, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


# ---------------------------------------------------------------------------
# molcore pipeline
# ---------------------------------------------------------------------------

def run_molcore(smiles: list[str]) -> dict[str, float]:
    from molcore.rdkit_bridge import standardize
    from molcore.pipeline import featurize_smiles
    from molcore.rdkit_bridge import calc_named_descriptors
    from molcore.io import MolDataset

    results: dict[str, float] = {}
    n = len(smiles)

    # 1. Standardize
    def _std():
        out = []
        for s in smiles:
            try:
                out.append(standardize(s))
            except Exception:
                out.append(s)
        return out

    std_smiles, t = timed("standardize", _std)
    results["standardize"] = t

    # 2. Fingerprints (Rust backend)
    fps, t = timed("fingerprints_rust", featurize_smiles, std_smiles, backend="rust")
    results["fingerprints_rust"] = t

    # 3. Descriptors (7 Lipinski)
    (arr, _), t = timed("descriptors", calc_named_descriptors, std_smiles, preset="lipinski")
    results["descriptors"] = t

    # 4. Scaffold split
    ds = MolDataset.from_smiles(std_smiles, compute_fps=False, compute_desc=False)
    _, t = timed("scaffold_split", ds.scaffold_split)
    results["scaffold_split"] = t

    # 5. PyG conversion
    def _pyg():
        from molcore.molecule import Mol
        graphs = []
        for s in std_smiles[:min(n, 200)]:
            try:
                graphs.append(Mol.from_smiles(s).to_pyg())
            except Exception:
                pass
        return graphs

    _, t = timed("pyg_conversion", _pyg)
    results["pyg_conversion_200"] = t

    results["total_mol"] = n
    results["total"] = sum(v for k, v in results.items() if k not in ("total_mol", "total"))
    return results


# ---------------------------------------------------------------------------
# RDKit baseline pipeline
# ---------------------------------------------------------------------------

def run_rdkit(smiles: list[str]) -> dict[str, float]:
    """
    Equivalent workflow using vanilla RDKit + manual PyG construction.
    Requires rdkit, torch-geometric.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    results: dict[str, float] = {}
    n = len(smiles)

    # 1. Standardize — three separate passes
    def _std():
        frag = rdMolStandardize.LargestFragmentChooser()
        uncharge = rdMolStandardize.Uncharger()
        taut = rdMolStandardize.TautomerEnumerator()
        out = []
        for s in smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    out.append(s)
                    continue
                mol = frag.choose(mol)
                mol = uncharge.uncharge(mol)
                mol = taut.Canonicalize(mol)
                out.append(Chem.MolToSmiles(mol))
            except Exception:
                out.append(s)
        return out

    std_smiles, t = timed("standardize", _std)
    results["standardize"] = t

    # 2. Fingerprints — per-molecule loop
    def _fps():
        rows = []
        for s in std_smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                arr = np.zeros(2048, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                rows.append(arr)
            except Exception:
                rows.append(np.zeros(2048, dtype=np.uint8))
        return np.stack(rows)

    _, t = timed("fingerprints", _fps)
    results["fingerprints_rdkit"] = t

    # 3. Descriptors — 7 columns
    _DESC_FNS = [
        ("MolWt",            Descriptors.MolWt),
        ("MolLogP",          Descriptors.MolLogP),
        ("NumHDonors",       Descriptors.NumHDonors),
        ("NumHAcceptors",    Descriptors.NumHAcceptors),
        ("TPSA",             Descriptors.TPSA),
        ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        ("RingCount",        Descriptors.RingCount),
    ]

    def _desc():
        rows = []
        for s in std_smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                rows.append([fn(mol) for _, fn in _DESC_FNS])
            except Exception:
                rows.append([np.nan] * len(_DESC_FNS))
        return np.array(rows, dtype=np.float32)

    _, t = timed("descriptors", _desc)
    results["descriptors"] = t

    # 4. Scaffold split (manual)
    def _split():
        from collections import defaultdict
        scaffold_map = defaultdict(list)
        for s in std_smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            except Exception:
                sc = s
            scaffold_map[sc].append(s)
        groups = sorted(scaffold_map.values(), key=len, reverse=True)
        n_train = int(len(std_smiles) * 0.8)
        n_val   = int(len(std_smiles) * 0.1)
        train, val, test = [], [], []
        for g in groups:
            if len(train) < n_train:
                train.extend(g)
            elif len(val) < n_val:
                val.extend(g)
            else:
                test.extend(g)
        return train, val, test

    _, t = timed("scaffold_split", _split)
    results["scaffold_split"] = t

    # 5. Manual PyG construction (subset of 200)
    def _pyg():
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            return []

        _ATOM_FEATURES = {
            "atomic_num": lambda a: a.GetAtomicNum(),
            "is_aromatic": lambda a: int(a.GetIsAromatic()),
            "formal_charge": lambda a: a.GetFormalCharge(),
            "num_hs": lambda a: a.GetTotalNumHs(),
        }
        graphs = []
        for s in std_smiles[:min(n, 200)]:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                node_feats = torch.tensor(
                    [[fn(a) for fn in _ATOM_FEATURES.values()] for a in mol.GetAtoms()],
                    dtype=torch.float32,
                )
                edges = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edges += [[i, j], [j, i]]
                edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
                graphs.append(Data(x=node_feats, edge_index=edge_index))
            except Exception:
                pass
        return graphs

    _, t = timed("pyg_conversion", _pyg)
    results["pyg_conversion_200"] = t

    results["total_mol"] = n
    results["total"] = sum(v for k, v in results.items() if k not in ("total_mol", "total"))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(n: int, mc: dict, rdk: dict | None) -> None:
    steps = [
        ("standardize",        "Standardize (strip salts, neutralize, tautomer)"),
        ("fingerprints_rust",  "ECFP4 fingerprints (Rust/molcore)"),
        ("fingerprints_rdkit", "ECFP4 fingerprints (RDKit baseline)"),
        ("descriptors",        "7 Lipinski descriptors"),
        ("scaffold_split",     "Scaffold train/val/test split"),
        ("pyg_conversion_200", "PyG Data objects (200 mols)"),
        ("total",              "TOTAL"),
    ]

    sep = "─" * 75
    print(f"\n{'End-to-End Benchmark':^75}")
    print(f"{'n = ' + str(n) + ' molecules':^75}")
    print(sep)
    hdr = f"{'Step':<45} {'molcore':>10}"
    if rdk:
        hdr += f"  {'RDKit':>10}  {'Speedup':>8}"
    print(hdr)
    print(sep)

    for key, label in steps:
        if key == "fingerprints_rust":
            mc_val = mc.get(key)
            rdk_val = rdk.get("fingerprints_rdkit") if rdk else None
        elif key == "fingerprints_rdkit":
            continue
        else:
            mc_val = mc.get(key)
            rdk_val = rdk.get(key) if rdk else None

        if mc_val is None:
            continue

        row = f"  {label:<43} {fmt(mc_val):>10}"
        if rdk and rdk_val:
            speedup = rdk_val / mc_val
            row += f"  {fmt(rdk_val):>10}  {speedup:>6.1f}×"
        print(row)

    print(sep)
    print(f"  {'Throughput (mol/s)  [fingerprints]':<43} {n / mc['fingerprints_rust']:>10,.0f}", end="")
    if rdk:
        print(f"  {n / rdk['fingerprints_rdkit']:>10,.0f}")
    else:
        print()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="molcore vs RDKit end-to-end benchmark")
    parser.add_argument("--n", type=int, default=1000, help="Number of molecules (default 1000)")
    parser.add_argument("--no-rdkit", action="store_true", help="Skip RDKit baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    smiles = make_smiles(args.n, seed=args.seed)
    print(f"Generated {len(smiles)} SMILES strings")

    print("Running molcore pipeline ...")
    mc_results = run_molcore(smiles)

    rdk_results = None
    if not args.no_rdkit:
        print("Running RDKit baseline ...")
        try:
            rdk_results = run_rdkit(smiles)
        except ImportError as e:
            print(f"  RDKit baseline skipped: {e}")

    print_report(args.n, mc_results, rdk_results)

    # Save results
    out = {
        "n": args.n,
        "molcore": mc_results,
        "rdkit": rdk_results,
    }
    results_path = Path(__file__).parent / "results_e2e.json"
    results_path.write_text(json.dumps(out, indent=2))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
