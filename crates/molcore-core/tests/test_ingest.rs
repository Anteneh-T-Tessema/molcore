use molcore::ingest::ingest;

#[test]
fn benzene_has_six_atoms_six_bonds() {
    let mol = ingest("c1ccccc1").expect("benzene should parse");
    assert_eq!(mol.graph.node_count(), 6, "benzene: 6 atoms");
    assert_eq!(mol.graph.edge_count(), 6, "benzene: 6 bonds");
}

#[test]
fn benzene_atoms_are_aromatic() {
    let mol = ingest("c1ccccc1").unwrap();
    for idx in mol.graph.node_indices() {
        assert!(mol.graph[idx].is_aromatic, "all benzene carbons should be aromatic");
    }
}

#[test]
fn ethanol_has_three_heavy_atoms() {
    let mol = ingest("CCO").expect("ethanol should parse");
    assert_eq!(mol.graph.node_count(), 3);
}

#[test]
fn acetic_acid_has_four_heavy_atoms() {
    let mol = ingest("CC(=O)O").expect("acetic acid should parse");
    assert_eq!(mol.graph.node_count(), 4);
}

#[test]
fn branch_connectivity_acetic_acid() {
    // CC(=O)O: C-C(=O)-O — 3 bonds total
    let mol = ingest("CC(=O)O").unwrap();
    assert_eq!(mol.graph.edge_count(), 3);
}

#[test]
fn invalid_smiles_returns_error() {
    assert!(ingest("NOT_A_SMILES").is_err(), "garbage SMILES must error");
}

#[test]
fn empty_string_returns_error() {
    assert!(ingest("").is_err());
}

#[test]
fn unclosed_ring_returns_error() {
    assert!(ingest("C1CC").is_err(), "unclosed ring bond must error");
}

#[test]
fn canonical_smiles_stored() {
    let mol = ingest("OCC").unwrap();
    assert!(!mol.canonical_smiles.is_empty());
}

#[test]
fn naphthalene_ring_system() {
    // c1ccc2ccccc2c1 — 10 atoms, 11 bonds (two fused 6-rings share one bond)
    let mol = ingest("c1ccc2ccccc2c1").expect("naphthalene should parse");
    assert_eq!(mol.graph.node_count(), 10);
    assert_eq!(mol.graph.edge_count(), 11);
}
