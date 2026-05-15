from molcore.molecule import Mol
from molcore.pipeline import featurize_smiles
from molcore.io import MolDataset, MolTorchDataset
from molcore.predictor import PropertyPredictor
from molcore import databases, gpu, pandas_tools
from molcore.rdkit_bridge import (
    # substructure
    filter_by_smarts,
    substructure_match,
    substructure_matches,
    # scaffold + clustering
    murcko_scaffold,
    scaffold_split,
    butina_cluster,
    # MMPA
    mmpa,
    # reactions
    react,
    react_bimolecular,
    enumerate_reactions,
    # standardization
    standardize,
    # MCS
    find_mcs,
    # R-groups
    rgroup_decompose,
    # additional fingerprints
    maccs_keys,
    atom_pairs_fp,
    topological_torsions_fp,
    rdkit_path_fp,
    # full descriptors
    calc_named_descriptors,
    list_descriptor_names,
    DESCRIPTOR_PRESETS,
    # SDF I/O
    write_sdf,
    mol_to_sdf_block,
    # depiction
    mol_to_svg,
    mol_to_png,
    mols_to_grid_svg,
)

__all__ = [
    # core
    "Mol",
    "featurize_smiles",
    "MolDataset",
    "MolTorchDataset",
    "PropertyPredictor",
    # substructure
    "filter_by_smarts",
    "substructure_match",
    "substructure_matches",
    # scaffold + clustering
    "murcko_scaffold",
    "scaffold_split",
    "butina_cluster",
    # MMPA
    "mmpa",
    # reactions
    "react",
    "react_bimolecular",
    "enumerate_reactions",
    # standardization
    "standardize",
    # analysis
    "find_mcs",
    "rgroup_decompose",
    # fingerprints
    "maccs_keys",
    "atom_pairs_fp",
    "topological_torsions_fp",
    "rdkit_path_fp",
    # descriptors
    "calc_named_descriptors",
    "list_descriptor_names",
    "DESCRIPTOR_PRESETS",
    # I/O
    "write_sdf",
    "mol_to_sdf_block",
    # depiction
    "mol_to_svg",
    "mol_to_png",
    "mols_to_grid_svg",
    # namespaces
    "databases",
    "gpu",
    "pandas_tools",
]
