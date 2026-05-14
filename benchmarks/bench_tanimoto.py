"""
Benchmark: Rust tanimoto_matrix vs RDKit BulkTanimoto.

Run with:
    python benchmarks/bench_tanimoto.py
    python benchmarks/bench_tanimoto.py --query 1000 --library 10000

Milestone 4 target: Rust >= 10x faster than RDKit on 10k x 10k.
"""
import argparse
import time
import random

POOL = [
    "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2ccccc2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1CCC[C@H]1c2cccnc2", "c1ccc(cc1)C(=O)O", "C1CCCCC1",
    "c1ccncc1", "CC(=O)N", "OC(=O)c1ccccc1O",
]


def bench_rust(query_smiles, library_smiles):
    from molcore.pipeline import featurize_smiles
    from molcore._molcore import tanimoto_matrix
    q_fps = featurize_smiles(query_smiles, backend="rust").numpy()
    l_fps = featurize_smiles(library_smiles, backend="rust").numpy()
    t0 = time.perf_counter()
    sim = tanimoto_matrix(q_fps, l_fps)
    return time.perf_counter() - t0, sim.shape


def bench_rdkit(query_smiles, library_smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np
    import time

    def fps(smiles):
        return [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
            for s in smiles
        ]

    q_fps = fps(query_smiles)
    l_fps = fps(library_smiles)

    t0 = time.perf_counter()
    for qfp in q_fps:
        DataStructs.BulkTanimotoSimilarity(qfp, l_fps)
    return time.perf_counter() - t0, (len(q_fps), len(l_fps))


def run(nq: int, nl: int):
    query   = [random.choice(POOL) for _ in range(nq)]
    library = [random.choice(POOL) for _ in range(nl)]

    print(f"\n{'='*60}")
    print(f"Tanimoto benchmark — {nq:,} queries × {nl:,} library")
    print(f"{'='*60}")

    # Warm up
    bench_rust(query[:5], library[:5])
    bench_rdkit(query[:5], library[:5])

    rust_t, shape = bench_rust(query, library)
    rdkit_t, _    = bench_rdkit(query, library)

    pairs = nq * nl
    print(f"  Rust  : {rust_t*1000:8.1f} ms  ({pairs/rust_t:>12,.0f} pairs/s)  shape={shape}")
    print(f"  RDKit : {rdkit_t*1000:8.1f} ms  ({pairs/rdkit_t:>12,.0f} pairs/s)")
    speedup = rdkit_t / rust_t
    print(f"  Speedup: {speedup:.1f}x")

    target = 10.0
    if speedup >= target:
        print(f"  PASS  — speedup {speedup:.1f}x >= target {target}x")
    else:
        print(f"  NOTE  — speedup {speedup:.1f}x < target {target}x  "
              f"(SIMD popcount path will be added to hit target on large matrices)")
    return speedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   type=int, default=100)
    parser.add_argument("--library", type=int, default=10_000)
    args = parser.parse_args()

    run(nq=50, nl=1_000)
    run(nq=args.query, nl=args.library)


if __name__ == "__main__":
    main()
