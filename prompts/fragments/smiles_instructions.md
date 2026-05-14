# Fragment: SMILES Handling Instructions

- Always canonicalize SMILES via `Mol.from_smiles(smi).smiles` before storing or comparing.
- Reject SMILES that fail RDKit sanitization. Surface the error to the user with the offending string.
- When extracting SMILES from free text, validate each candidate before use.
- Do not attempt to fix invalid SMILES — reject and ask for correction.
