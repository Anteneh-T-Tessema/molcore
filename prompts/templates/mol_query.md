# Template: Molecular Query

Given the molecule `{{smiles}}`:

1. Compute ECFP4 fingerprint (radius={{radius}}, nbits={{nbits}}, backend={{backend}})
2. Report: shape, sparsity (fraction of 1-bits), first 10 set bits
3. If `library_smiles` is provided, return top {{top_k}} similar compounds by Tanimoto score
