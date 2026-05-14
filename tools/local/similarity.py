"""
Tool: similarity
Strict function signature for agent tool-use.
"""
from __future__ import annotations
import numpy as np
from molcore.pipeline import featurize_smiles
from molcore._molcore import tanimoto_matrix


def run(query_smiles: list[str], library_smiles: list[str], top_k: int = 10) -> dict:
    q_fps = featurize_smiles(query_smiles).numpy()
    l_fps = featurize_smiles(library_smiles).numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)  # (Q, L)

    results = []
    for qi, qsmi in enumerate(query_smiles):
        row = sim[qi].numpy()
        ranked = sorted(
            zip(library_smiles, row.tolist()),
            key=lambda x: x[1], reverse=True
        )[:top_k]
        results.append({"query": qsmi, "hits": [{"smiles": s, "tanimoto": round(t, 4)} for s, t in ranked]})

    return {"results": results}
