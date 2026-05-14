# Skill: reaction

## When to invoke
When an agent needs to apply chemical transformations: metabolite prediction, protecting-group
removal, prodrug activation, combinatorial library synthesis, or any rule-based reaction.

## API overview

### Unimolecular transform — `react`
Apply a reaction SMARTS to a single molecule.

```python
from molcore.rdkit_bridge import react

products = react("CC(=O)OCC", "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]")
# returns deduplicated list of product SMILES
```

| Arg | Type | Description |
|---|---|---|
| `smiles` | `str` | Reactant SMILES |
| `rxn_smarts` | `str` | Reaction SMARTS (`reactants>>products`) |

Returns `list[str]` — sorted, deduplicated product SMILES.  
Returns `[]` if no pattern matches; raises `ValueError` on invalid SMARTS.

### Bimolecular transform — `react_bimolecular`
```python
from molcore.rdkit_bridge import react_bimolecular

products = react_bimolecular(
    "CC(=O)O",             # acid
    "CCN",                 # amine
    "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]",  # amide coupling
)
```

### Library enumeration — `enumerate_reactions`
Apply a transform to every molecule in a list; collect all unique products.

```python
from molcore.rdkit_bridge import enumerate_reactions

esters = ["CC(=O)OCC", "CC(=O)OCCC", "CC(=O)OC"]
hydrolysis = "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]"
all_products = enumerate_reactions(esters, hydrolysis, max_products=500)
```

| Arg | Type | Default | Description |
|---|---|---|---|
| `reactants` | `list[str]` | — | Library of reactant SMILES |
| `rxn_smarts` | `str` | — | Reaction SMARTS |
| `max_products` | `int` | `1000` | Hard cap on returned products |

### Mol-level API
```python
from molcore.molecule import Mol

mol = Mol.from_smiles("CC(=O)OCC")
products: list[Mol] = mol.react("[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]")
```
Returns `list[Mol]` — each product is a fully-parsed `Mol` object.

## Common reaction SMARTS patterns

| Transform | SMARTS |
|---|---|
| Ester hydrolysis | `[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]` |
| Amide coupling | `[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]` |
| N-Boc deprotection | `[N:1][C](=O)OC(C)(C)C>>[N:1]` |
| Amine alkylation | `[N:1].[C:2][Br]>>[N:1][C:2]` |
| Reductive amination | `[N:1].[C:2]=O>>[N:1][C:2]` |

## Error handling
- `ValueError("Invalid reaction SMARTS: ...")` — raised on malformed SMARTS
- Invalid reactant SMILES raise `ValueError` (propagated from `from_smiles`)
- Non-matching reactants silently produce empty product lists

## When NOT to use
- Do not use for quantum-mechanical accuracy — SMARTS encodes substructural rules only.
- Stereochemistry handling at reaction centers is approximate; verify stereocenters with RDKit or a 3D engine.
- `enumerate_reactions` with very large libraries (>10 k) may be slow; prefer batching or the `max_products` cap.
