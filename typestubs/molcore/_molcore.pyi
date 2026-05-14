"""
Type stubs for molcore._molcore (Rust/PyO3 extension).
Pyright reads these before the extension is built, eliminating import errors.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

NODE_FEAT_DIM: int  # = 9

# Node feature layout (mol_to_graph_arrays column indices):
#   0  atomic_num       — raw atomic number
#   1  is_aromatic      — 0.0 or 1.0
#   2  formal_charge    — signed integer as float
#   3  num_hs           — implicit hydrogen count
#   4  degree           — heavy-atom neighbor count
#   5  in_ring          — 0.0 or 1.0 (BFS ring detection)
#   6  hybridization    — 0=unknown, 1=sp, 2=sp2, 3=sp3
#   7  chirality        — 0=none, 1=@ (S), 2=@@ (R)
#   8  mass_norm        — atomic_mass / 100.0


class PyMolGraph:
    @staticmethod
    def from_smiles_rdkit(smiles: str) -> PyMolGraph: ...
    def canonical_smiles(self) -> str: ...
    def num_atoms(self) -> int: ...
    def num_bonds(self) -> int: ...


def ecfp4_batch(
    smiles_list: list[str],
    radius: int = 2,
    nbits: int = 2048,
) -> NDArray[np.uint8]:
    """
    Batch ECFP4 fingerprints via Rayon-parallel Rust.
    Returns (N, nbits) uint8 numpy array — zero-copy transfer via IntoPyArray.
    """
    ...


def mol_to_graph_arrays(
    mol: PyMolGraph,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]:
    """
    Extract graph arrays from a PyMolGraph.

    Returns:
        node_feats : (N, 9)  float32  — see NODE_FEAT_DIM layout above
        edge_index : (2, E)  int64    — COO bidirectional, suitable for PyG
        edge_feats : (E, 4)  float32  — bond one-hot [single, double, triple, aromatic]

    All arrays are zero-copy (IntoPyArray / torch.from_numpy shares the buffer).
    """
    ...


def tanimoto_matrix(
    query: NDArray[np.uint8],
    library: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """
    Pairwise Tanimoto similarity — Rayon-parallel u64 popcount.

    Args:
        query   : (Q, B) uint8
        library : (L, B) uint8
    Returns:
        (Q, L) float32 — values in [0, 1]
    """
    ...


def calc_descriptors_batch(
    smiles_list: list[str],
) -> NDArray[np.float32]:
    """
    Batch molecular descriptors via Rust approximation.

    Returns (N, 3) float32:
        col 0 : MW (exact atomic masses)
        col 1 : LogP (Crippen fragment approximation)
        col 2 : heavy atom count

    For exact TPSA and Crippen LogP use rdkit_bridge.calc_descriptors_rdkit().
    """
    ...
