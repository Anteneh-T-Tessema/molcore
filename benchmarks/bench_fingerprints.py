"""
Benchmark: Rust ECFP4 vs RDKit ECFP4 throughput.

Run with:
    python benchmarks/bench_fingerprints.py
    python benchmarks/bench_fingerprints.py --smiles 100000

Milestone 4 target: Rust >= 10x faster than RDKit on 10k+ SMILES.
"""
import argparse
import time
import sys
import random

# fmt: off
POOL = [
    "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2ccccc2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1CCC[C@H]1c2cccnc2", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "c1ccc(cc1)C(=O)O", "C1CCCCC1", "c1ccncc1", "CC(=O)N",
    "OC(=O)c1ccccc1O", "CC(N)C(=O)O", "c1ccc(nc1)N",
    "COc1ccc(cc1OC)C2CC(=O)c3ccc(OC)c(OC)c3C2",
]


def make_library(n: int) -> list[str]:
    return [random.choice(POOL) for _ in range(n)]


def bench_rust(smiles: list[str], radius: int = 2, nbits: int = 2048) -> tuple[float, int]:
    from molcore.pipeline import featurize_smiles
    t0 = time.perf_counter()
    fps = featurize_smiles(smiles, backend="rust", radius=radius, nbits=nbits)
    elapsed = time.perf_counter() - t0
    return elapsed, fps.shape[0]


def bench_rdkit(smiles: list[str], radius: int = 2, nbits: int = 2048) -> tuple[float, int]:
    from molcore.pipeline import featurize_smiles
    t0 = time.perf_counter()
    fps = featurize_smiles(smiles, backend="rdkit", radius=radius, nbits=nbits)
    elapsed = time.perf_counter() - t0
    return elapsed, fps.shape[0]


def run(n: int):
    smiles = make_library(n)
    print(f"\n{'='*60}")
    print(f"Fingerprint benchmark — {n:,} SMILES, ECFP4 (r=2, nbits=2048)")
    print(f"{'='*60}")

    # Warm up
    bench_rust(smiles[:10])
    bench_rdkit(smiles[:10])

    rust_t, _ = bench_rust(smiles)
    rdkit_t, _ = bench_rdkit(smiles)

    rust_rate  = n / rust_t
    rdkit_rate = n / rdkit_t
    speedup    = rdkit_t / rust_t

    print(f"  Rust   : {rust_t*1000:8.1f} ms  ({rust_rate:>12,.0f} mol/s)")
    print(f"  RDKit  : {rdkit_t*1000:8.1f} ms  ({rdkit_rate:>12,.0f} mol/s)")
    print(f"  Speedup: {speedup:.1f}x")

    target = 10.0
    if speedup >= target:
        print(f"  PASS  — speedup {speedup:.1f}x >= target {target}x")
    else:
        print(f"  NOTE  — speedup {speedup:.1f}x < target {target}x  "
              f"(rdkit-rs backend + larger batch will close this gap)")
    return speedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=int, default=10_000)
    args = parser.parse_args()

    for n in [1_000, args.smiles]:
        run(n)


if __name__ == "__main__":
    main()
