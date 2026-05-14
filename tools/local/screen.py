"""
Tool: screen — full virtual screening pipeline.
"""
from __future__ import annotations
from molcore.pipeline import featurize_smiles
from molcore._molcore import tanimoto_matrix


def run(
    query_smiles: list[str],
    library_smiles: list[str],
    threshold: float = 0.3,
    top_k: int = 50,
) -> dict:
    q_fps = featurize_smiles(query_smiles).numpy()
    l_fps = featurize_smiles(library_smiles).numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)

    hits = []
    for qi, qsmi in enumerate(query_smiles):
        row = sim[qi].numpy()
        for li, (lsmi, score) in enumerate(zip(library_smiles, row.tolist())):
            if score >= threshold:
                hits.append({"query": qsmi, "hit": lsmi, "tanimoto": round(score, 4)})

    hits.sort(key=lambda x: x["tanimoto"], reverse=True)
    return {"total_screened": len(library_smiles), "hits_above_threshold": len(hits), "top_hits": hits[:top_k]}
