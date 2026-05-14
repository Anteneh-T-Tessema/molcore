# Template: Virtual Screening Query

Screen `{{library_size}}` compounds against query `{{query_smiles}}`.

Pipeline:
1. Fingerprint query: `featurize_smiles([query], backend="rust")`
2. Fingerprint library: `featurize_smiles(library, backend="rust")`
3. Tanimoto matrix: `tanimoto_matrix(query_fp, library_fp)`
4. Filter: Tanimoto ≥ {{threshold}}
5. Return top {{top_k}} hits with scores

Report runtime and throughput (compounds/sec).
