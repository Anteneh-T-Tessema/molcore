# Semantic Memory: Chemistry Knowledge

## Fingerprint conventions
- ECFP4 = Morgan radius 2, 2048 bits. Standard for virtual screening.
- ECFP6 = Morgan radius 3. Better for scaffold hopping.
- RDKit and Rust ECFP4 use different hash seeds → bit vectors are NOT interchangeable.

## Tanimoto thresholds (rule of thumb)
- ≥ 0.85: very similar (likely same scaffold)
- 0.65–0.85: similar (possible bioisostere)
- < 0.40: structurally diverse

## Lipinski Ro5
- MW ≤ 500, LogP ≤ 5, HBA ≤ 10, HBD ≤ 5
- Use rdkit backend for exact Ro5 assessment

## Data sources
- PubChem: structure lookup, synonyms, basic properties
- ChEMBL: bioactivity (IC50, Ki, Kd) — quality-filtered experimental data
- ZINC: purchasable compound libraries for screening
