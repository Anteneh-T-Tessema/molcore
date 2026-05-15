from __future__ import annotations
from dataclasses import dataclass
from molcore._molcore import PyMolGraph
from molcore import rdkit_bridge


@dataclass(frozen=True)
class Mol:
    """
    Immutable molecule. The Rust MolGraph is the canonical representation.
    RDKit Mol is derived on demand and never stored — calling rdmol() twice
    rebuilds it. This is intentional: no cached mutable state.

    Transformations return NEW Mol instances — this Mol is never modified.
    """
    _graph: PyMolGraph
    smiles: str  # canonical SMILES — set at construction, never changes

    @classmethod
    def from_smiles(cls, smiles: str) -> "Mol":
        graph = PyMolGraph.from_smiles_rdkit(smiles)
        canonical = graph.canonical_smiles()
        return cls(_graph=graph, smiles=canonical)

    @classmethod
    def from_molblock(cls, molblock: str) -> "Mol":
        """Parse an MDL Mol block string into a Mol. Raises ValueError on failure."""
        from rdkit import Chem as _Chem
        rdmol = rdkit_bridge.from_molblock(molblock)
        return cls.from_smiles(rdkit_bridge.canonicalize(_Chem.MolToSmiles(rdmol)))

    def rdmol(self):
        """Fresh RDKit Mol from canonical SMILES. Never cached — callers own it."""
        return rdkit_bridge.from_smiles(self.smiles)

    # --- transformations: always return new Mol, never mutate ---

    def neutralize(self) -> "Mol":
        return Mol.from_smiles(rdkit_bridge.neutralize(self.smiles))

    def strip_salts(self) -> "Mol":
        return Mol.from_smiles(rdkit_bridge.strip_salts(self.smiles))

    def standardize(self) -> "Mol":
        """Full standardization: strip salts → neutralize → canonical tautomer."""
        return Mol.from_smiles(rdkit_bridge.standardize(self.smiles))

    # --- featurization delegates to Rust batch API ---

    def to_pyg(self):
        from molcore.featurizers.graph import to_pyg_data
        return to_pyg_data(self._graph)

    def to_dgl(self):
        from molcore.featurizers.graph import to_dgl_graph
        return to_dgl_graph(self._graph)

    def to_pyg_hetero(self):
        from molcore.featurizers.graph import to_pyg_hetero
        return to_pyg_hetero(self._graph)

    def fingerprint(self, radius: int = 2, nbits: int = 2048, backend: str = "rust"):
        from molcore.pipeline import featurize_smiles
        return featurize_smiles([self.smiles], radius=radius, nbits=nbits, backend=backend)[0]

    # --- substructure search ---

    def matches(self, smarts: str) -> bool:
        """Return True if this molecule contains the SMARTS substructure."""
        return rdkit_bridge.substructure_match(self.smiles, smarts)

    def find_substructures(self, smarts: str) -> list[tuple[int, ...]]:
        """Return all atom-index tuples matching the SMARTS pattern."""
        return rdkit_bridge.substructure_matches(self.smiles, smarts)

    # --- reaction transforms ---

    def react(self, rxn_smarts: str) -> list["Mol"]:
        """
        Apply a unimolecular reaction SMARTS to this molecule.

        Returns a list of product Mol instances (empty list if no reaction occurs).
        Example — methyl ester hydrolysis:
            mol.react("[C:1](=O)O[CH3:2]>>[C:1](=O)[OH]")
        """
        products = rdkit_bridge.react(self.smiles, rxn_smarts)
        result = []
        for smi in products:
            try:
                result.append(Mol.from_smiles(smi))
            except Exception:
                pass
        return result

    # --- scaffold ---

    def scaffold(self, generic: bool = False) -> "Mol":
        """Return the Murcko scaffold as a new Mol (empty Mol if no rings)."""
        smi = rdkit_bridge.murcko_scaffold(self.smiles, generic=generic)
        if not smi:
            return Mol.from_smiles("C")  # degenerate: no ring system
        return Mol.from_smiles(smi)

    # --- 3D conformers ---

    def conformers(
        self,
        n_confs: int = 1,
        seed: int = 42,
        force_field: str = "MMFF94",
    ):
        """
        Generate 3D conformers. Returns list of (n_atoms, 3) numpy arrays.
        Requires: rdkit (always available) + a molecule that can be embedded.
        """
        import numpy as np
        return rdkit_bridge.generate_conformers(
            self.smiles, n_confs=n_confs, seed=seed, force_field=force_field
        )

    def descriptors_3d(self, seed: int = 42) -> dict[str, float]:
        """Compute 3D shape descriptors (PMI, asphericity, etc.)."""
        return rdkit_bridge.calc_descriptors_3d(self.smiles, seed=seed)

    # --- 2D depiction ---

    def to_svg(
        self,
        width: int = 300,
        height: int = 200,
        highlight_atoms: "list[int] | None" = None,
        highlight_bonds: "list[int] | None" = None,
    ) -> str:
        """Render this molecule as an SVG string."""
        return rdkit_bridge.mol_to_svg(
            self.smiles, width=width, height=height,
            highlight_atoms=highlight_atoms, highlight_bonds=highlight_bonds,
        )

    def to_png(self, path: str, width: int = 300, height: int = 200) -> None:
        """Render this molecule to a PNG file."""
        rdkit_bridge.mol_to_png(self.smiles, path=path, width=width, height=height)

    def _repr_svg_(self) -> str:
        """Jupyter SVG display."""
        return self.to_svg(width=300, height=200)

    def _repr_html_(self) -> str:
        """Jupyter HTML display (wraps SVG in a div with SMILES tooltip)."""
        svg = self.to_svg(width=300, height=200)
        return (
            f'<div title="{self.smiles}" style="display:inline-block">{svg}</div>'
        )

    def __repr__(self) -> str:
        return f"Mol(smiles={self.smiles!r}, atoms={self._graph.num_atoms()}, bonds={self._graph.num_bonds()})"
