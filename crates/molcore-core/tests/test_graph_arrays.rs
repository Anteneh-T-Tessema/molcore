use molcore::graph_arrays::{mol_to_graph_arrays_raw, NODE_FEAT_DIM};
use molcore::ingest::ingest;

// ── node feature shapes ─────────────────────────────────────────────────────

#[test]
fn methane_node_shape() {
    let mol = ingest("C").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[1, NODE_FEAT_DIM], "methane: 1 atom");
    assert_eq!(ei.shape(), &[2, 0], "methane: no bonds");
    assert_eq!(ef.shape(), &[0, 4], "methane: no edge features");
}

#[test]
fn ethanol_node_shape() {
    let mol = ingest("CCO").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[3, NODE_FEAT_DIM]);
    assert_eq!(ei.shape(), &[2, 4]);
    assert_eq!(ef.shape(), &[4, 4]);
}

#[test]
fn benzene_edge_count() {
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[6, NODE_FEAT_DIM], "benzene: 6 carbons");
    assert_eq!(ei.shape(), &[2, 12], "benzene: 6 bonds × 2 directions");
    assert_eq!(ef.shape(), &[12, 4]);
}

// ── node feature values: original 4 ─────────────────────────────────────────

#[test]
fn methane_node_feature_values() {
    let mol = ingest("C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 0]], 6.0,  "atomic_num carbon");
    assert_eq!(nf[[0, 1]], 0.0,  "not aromatic");
    assert_eq!(nf[[0, 2]], 0.0,  "no formal charge");
    assert_eq!(nf[[0, 3]], 4.0,  "methane: 4 implicit H");
}

#[test]
fn benzene_aromatic_flag() {
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    for i in 0..6 {
        assert_eq!(nf[[i, 1]], 1.0, "benzene C[{i}] must be aromatic");
    }
}

#[test]
fn nitrogen_atomic_num() {
    let mol = ingest("c1ccncc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[3, 0]], 7.0, "pyridine N atomic_num");
    assert_eq!(nf[[3, 1]], 1.0, "pyridine N is aromatic");
}

// ── degree (feat 4) ─────────────────────────────────────────────────────────

#[test]
fn methane_degree_zero() {
    let mol = ingest("C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 4]], 0.0, "methane: degree 0 (no heavy neighbors)");
}

#[test]
fn ethanol_degrees() {
    // CCO: CH3(1) — CH2(2) — OH(1)
    let mol = ingest("CCO").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 4]], 1.0, "terminal C: degree 1");
    assert_eq!(nf[[1, 4]], 2.0, "middle C: degree 2");
    assert_eq!(nf[[2, 4]], 1.0, "O: degree 1");
}

#[test]
fn benzene_degree_two() {
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    for i in 0..6 {
        assert_eq!(nf[[i, 4]], 2.0, "benzene C[{i}]: degree 2");
    }
}

// ── in_ring (feat 5) ────────────────────────────────────────────────────────

#[test]
fn methane_not_in_ring() {
    let mol = ingest("C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 5]], 0.0, "methane not in ring");
}

#[test]
fn benzene_all_in_ring() {
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    for i in 0..6 {
        assert_eq!(nf[[i, 5]], 1.0, "benzene C[{i}] must be in_ring");
    }
}

#[test]
fn toluene_methyl_not_in_ring() {
    // Cc1ccccc1: methyl C (idx 0) is NOT in ring
    let mol = ingest("Cc1ccccc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 5]], 0.0, "methyl C not in ring");
    // ring carbons (idx 1-6) must be in ring
    for i in 1..7 { assert_eq!(nf[[i, 5]], 1.0, "ring C[{i}] in ring"); }
}

// ── hybridization (feat 6) ──────────────────────────────────────────────────

#[test]
fn benzene_sp2() {
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    for i in 0..6 { assert_eq!(nf[[i, 6]], 2.0, "benzene C sp2"); }
}

#[test]
fn ethane_sp3() {
    let mol = ingest("CC").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 6]], 3.0, "ethane C sp3");
    assert_eq!(nf[[1, 6]], 3.0, "ethane C sp3");
}

#[test]
fn ethylene_sp2() {
    let mol = ingest("C=C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 6]], 2.0, "ethylene C sp2");
}

#[test]
fn acetylene_sp() {
    let mol = ingest("C#C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 6]], 1.0, "acetylene C sp");
}

// ── chirality (feat 7) ──────────────────────────────────────────────────────

#[test]
fn achiral_carbon_zero() {
    let mol = ingest("CCO").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[0, 7]], 0.0, "no chirality");
}

#[test]
fn bracket_chiral_at() {
    // L-alanine: [C@H] is the chiral center
    let mol = ingest("N[C@H](C)C(=O)O").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    // atom 1 is [C@H]: chirality = 1
    assert_eq!(nf[[1, 7]], 1.0, "[C@H] chirality = 1");
}

#[test]
fn bracket_chiral_atat() {
    let mol = ingest("N[C@@H](C)C(=O)O").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf[[1, 7]], 2.0, "[C@@H] chirality = 2");
}

// ── mass_normalized (feat 8) ────────────────────────────────────────────────

#[test]
fn carbon_mass_norm() {
    let mol = ingest("C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    let expected = 12.011_f32 / 100.0;
    assert!((nf[[0, 8]] - expected).abs() < 1e-4, "C mass_norm");
}

#[test]
fn chlorine_mass_norm() {
    let mol = ingest("CCl").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    let expected = 35.45_f32 / 100.0;
    assert!((nf[[1, 8]] - expected).abs() < 1e-4, "Cl mass_norm");
}

// ── edge index and one-hot encoding ─────────────────────────────────────────

#[test]
fn edge_index_bidirectional() {
    let mol = ingest("CC").unwrap();
    let (_, ei, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(ei.shape(), &[2, 2]);
    let pairs: Vec<(i64, i64)> = (0..2).map(|i| (ei[[0, i]], ei[[1, i]])).collect();
    assert!(pairs.contains(&(0, 1)));
    assert!(pairs.contains(&(1, 0)));
}

#[test]
fn single_bond_one_hot()   { let m = ingest("CC").unwrap();  let (_, _, ef) = mol_to_graph_arrays_raw(&m); assert_eq!(&ef.row(0).to_vec(), &[1.,0.,0.,0.]); }
#[test]
fn double_bond_one_hot()   { let m = ingest("C=C").unwrap(); let (_, _, ef) = mol_to_graph_arrays_raw(&m); assert_eq!(&ef.row(0).to_vec(), &[0.,1.,0.,0.]); }
#[test]
fn triple_bond_one_hot()   { let m = ingest("C#C").unwrap(); let (_, _, ef) = mol_to_graph_arrays_raw(&m); assert_eq!(&ef.row(0).to_vec(), &[0.,0.,1.,0.]); }
#[test]
fn aromatic_bond_one_hot() { let m = ingest("c1ccccc1").unwrap(); let (_, _, ef) = mol_to_graph_arrays_raw(&m); assert_eq!(&ef.row(0).to_vec(), &[0.,0.,0.,1.]); }
