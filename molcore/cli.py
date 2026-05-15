"""
molcore CLI — featurize, screen, benchmark.

Usage:
    molcore featurize  [options] smiles.txt
    molcore screen     [options] query.smi library.smi
    molcore benchmark  [options]
"""
from __future__ import annotations

import argparse
import sys
import time


# ---------------------------------------------------------------------------
# featurize
# ---------------------------------------------------------------------------

def cmd_featurize(args: argparse.Namespace) -> None:
    """Batch-featurize SMILES from a file, write fingerprints as .npy."""
    import numpy as np
    from molcore.pipeline import featurize_smiles

    with open(args.input) as fh:
        smiles = [line.strip() for line in fh if line.strip()]

    print(f"Featurizing {len(smiles)} molecules (backend={args.backend}, nbits={args.nbits})…")
    t0 = time.perf_counter()
    fps = featurize_smiles(smiles, backend=args.backend, nbits=args.nbits)
    elapsed = time.perf_counter() - t0

    out_path = args.output or args.input.rsplit(".", 1)[0] + "_fps.npy"
    np.save(out_path, fps.numpy())
    rate = len(smiles) / elapsed
    print(f"Saved {fps.shape} fingerprints → {out_path}")
    print(f"Throughput: {rate:,.0f} mol/s  ({elapsed:.2f}s total)")


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

def cmd_screen(args: argparse.Namespace) -> None:
    """
    Virtual screen: rank library molecules by Tanimoto similarity to query.

    Outputs a TSV: rank, smiles, similarity_score.
    """
    import numpy as np
    from molcore.pipeline import featurize_smiles
    from molcore._molcore import tanimoto_matrix

    def _read(path: str) -> list[str]:
        with open(path) as fh:
            return [line.split()[0] for line in fh if line.strip()]

    queries  = _read(args.query)
    library  = _read(args.library)

    print(f"Query: {len(queries)} mol(s)  Library: {len(library)} mol(s)")
    q_fps = featurize_smiles(queries,  backend=args.backend).numpy()
    l_fps = featurize_smiles(library, backend=args.backend).numpy()

    sim = tanimoto_matrix(q_fps, l_fps)  # (Q, L) float32

    # Aggregate: max similarity across queries for each library molecule
    scores = sim.max(axis=0)
    order  = scores.argsort()[::-1]

    out = args.output or "screen_results.tsv"
    with open(out, "w") as fh:
        fh.write("rank\tsmiles\tsimilarity\n")
        for rank, idx in enumerate(order[: args.top_k], start=1):
            fh.write(f"{rank}\t{library[idx]}\t{scores[idx]:.4f}\n")

    print(f"Top-{args.top_k} results → {out}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark Rust vs RDKit fingerprint throughput."""
    import numpy as np
    from molcore.pipeline import featurize_smiles

    # Built-in set of drug-like SMILES for reproducibility
    BENCH_SMILES = [
        "CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
        "CC(=O)Nc1ccc(cc1)O",
    ] * (args.n // 10 + 1)
    bench = BENCH_SMILES[: args.n]

    print(f"Benchmark: {len(bench)} SMILES × {args.repeats} repeats")
    print(f"{'Backend':<10} {'mol/s':>10}  {'ms/mol':>8}")

    for backend in ["rust", "rdkit"]:
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            featurize_smiles(bench, backend=backend, nbits=2048)
            times.append(time.perf_counter() - t0)
        best = min(times)
        rate = len(bench) / best
        print(f"{backend:<10} {rate:>10,.0f}  {best / len(bench) * 1000:>8.3f}")


# ---------------------------------------------------------------------------
# admet-screen
# ---------------------------------------------------------------------------

def cmd_admet_screen(args: argparse.Namespace) -> None:
    """Run rule-based ADMET profiling on a SMILES file, write TSV."""
    from molcore.admet import admet_screen_df

    with open(args.input) as fh:
        smiles = [line.strip() for line in fh if line.strip()]

    print(f"Screening {len(smiles)} molecules…")
    df = admet_screen_df(smiles)

    out = args.output or args.input.rsplit(".", 1)[0] + "_admet.tsv"
    df.to_csv(out, sep="\t", index=False)

    passing = int(df["druglike"].sum()) if "druglike" in df.columns else "?"
    print(f"Drug-like: {passing}/{len(df)}  →  {out}")


# ---------------------------------------------------------------------------
# scaffold-split (bonus subcommand)
# ---------------------------------------------------------------------------

def cmd_scaffold_split(args: argparse.Namespace) -> None:
    """Scaffold-aware train/val/test split of a SMILES file."""
    from molcore.rdkit_bridge import scaffold_split

    with open(args.input) as fh:
        smiles = [line.strip() for line in fh if line.strip()]

    train, val, test = scaffold_split(
        smiles,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    def _write(path: str, lst: list[str]) -> None:
        with open(path, "w") as fh:
            fh.write("\n".join(lst) + "\n")

    stem = args.input.rsplit(".", 1)[0]
    _write(args.train_out or f"{stem}_train.smi",  train)
    _write(args.val_out   or f"{stem}_val.smi",    val)
    _write(args.test_out  or f"{stem}_test.smi",   test)
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molcore",
        description="AI-native cheminformatics toolkit",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # featurize
    p = sub.add_parser("featurize", help="Batch ECFP4 fingerprints from a SMILES file")
    p.add_argument("input",   help="Input SMILES file (one SMILES per line)")
    p.add_argument("-o", "--output", default=None, help="Output .npy path (default: <input>_fps.npy)")
    p.add_argument("--backend", choices=["rust", "rdkit"], default="rust")
    p.add_argument("--nbits", type=int, default=2048)
    p.set_defaults(func=cmd_featurize)

    # screen
    p = sub.add_parser("screen", help="Virtual screen a library against queries by Tanimoto")
    p.add_argument("query",   help="Query SMILES file")
    p.add_argument("library", help="Library SMILES file")
    p.add_argument("-o", "--output", default=None, help="Output TSV path")
    p.add_argument("--top-k", type=int, default=100, dest="top_k")
    p.add_argument("--backend", choices=["rust", "rdkit"], default="rust")
    p.set_defaults(func=cmd_screen)

    # benchmark
    p = sub.add_parser("benchmark", help="Benchmark Rust vs RDKit fingerprint throughput")
    p.add_argument("-n", type=int, default=1000, help="Number of molecules (default 1000)")
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=cmd_benchmark)

    # admet-screen
    p = sub.add_parser("admet-screen", help="Rule-based ADMET profiling of a SMILES file")
    p.add_argument("input", help="Input SMILES file (one SMILES per line)")
    p.add_argument("-o", "--output", default=None, help="Output TSV path (default: <input>_admet.tsv)")
    p.set_defaults(func=cmd_admet_screen)

    # scaffold-split
    p = sub.add_parser("scaffold-split", help="Scaffold-aware train/val/test split")
    p.add_argument("input",  help="Input SMILES file")
    p.add_argument("--train-frac", type=float, default=0.8, dest="train_frac")
    p.add_argument("--val-frac",   type=float, default=0.1, dest="val_frac")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--train-out",  default=None, dest="train_out")
    p.add_argument("--val-out",    default=None, dest="val_out")
    p.add_argument("--test-out",   default=None, dest="test_out")
    p.set_defaults(func=cmd_scaffold_split)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
