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

    def rdmol(self):
        """Fresh RDKit Mol from canonical SMILES. Never cached — callers own it."""
        return rdkit_bridge.from_smiles(self.smiles)

    # --- transformations: always return new Mol, never mutate ---

    def neutralize(self) -> "Mol":
        return Mol.from_smiles(rdkit_bridge.neutralize(self.smiles))

    def strip_salts(self) -> "Mol":
        return Mol.from_smiles(rdkit_bridge.strip_salts(self.smiles))

    # --- featurization delegates to Rust batch API ---

    def to_pyg(self):
        from molcore.featurizers.graph import to_pyg_data
        return to_pyg_data(self._graph)

    def to_dgl(self):
        from molcore.featurizers.graph import to_dgl_graph
        return to_dgl_graph(self._graph)

    def fingerprint(self, radius: int = 2, nbits: int = 2048, backend: str = "rust"):
        from molcore.pipeline import featurize_smiles
        return featurize_smiles([self.smiles], radius=radius, nbits=nbits, backend=backend)[0]

    def __repr__(self) -> str:
        return f"Mol(smiles={self.smiles!r}, atoms={self._graph.num_atoms()}, bonds={self._graph.num_bonds()})"
