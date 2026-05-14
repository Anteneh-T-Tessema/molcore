"""
Type stubs for molcore._molcore (Rust/PyO3 extension).
Pyright reads these before the extension is built, eliminating import errors.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


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
) -> NDArray[np.uint8]: ...


def mol_to_graph_arrays(
    mol: PyMolGraph,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]: ...


def tanimoto_matrix(
    query: NDArray[np.uint8],
    library: NDArray[np.uint8],
) -> NDArray[np.float32]: ...


def calc_descriptors_batch(
    smiles_list: list[str],
) -> NDArray[np.float32]: ...
