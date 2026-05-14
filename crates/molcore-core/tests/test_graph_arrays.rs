use molcore::graph_arrays::mol_to_graph_arrays_raw;
use molcore::ingest::ingest;

// mol_to_graph_arrays_raw returns (node_feats, edge_index, edge_feats) as plain
// ndarray::Array2 values so they can be tested without a Python runtime.

// ── node feature shapes ─────────────────────────────────────────────────────

#[test]
fn methane_node_shape() {
    let mol = ingest("C").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[1, 4], "methane: 1 atom, 4 features");
    assert_eq!(ei.shape(), &[2, 0], "methane: no bonds");
    assert_eq!(ef.shape(), &[0, 4], "methane: no edge features");
}

#[test]
fn ethanol_node_shape() {
    // CCO: 3 heavy atoms, 2 bonds → 4 directed edges
    let mol = ingest("CCO").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[3, 4]);
    assert_eq!(ei.shape(), &[2, 4]);
    assert_eq!(ef.shape(), &[4, 4]);
}

#[test]
fn benzene_edge_count() {
    // 6 aromatic bonds → 12 directed edges
    let mol = ingest("c1ccccc1").unwrap();
    let (nf, ei, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(nf.shape(), &[6, 4], "benzene: 6 carbons");
    assert_eq!(ei.shape(), &[2, 12], "benzene: 6 bonds × 2 directions");
    assert_eq!(ef.shape(), &[12, 4]);
}

// ── node feature values ─────────────────────────────────────────────────────

#[test]
fn methane_node_feature_values() {
    let mol = ingest("C").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    // [atomic_num=6, is_aromatic=0, formal_charge=0, num_hs=4]
    assert_eq!(nf[[0, 0]], 6.0,  "atomic_num of carbon");
    assert_eq!(nf[[0, 1]], 0.0,  "not aromatic");
    assert_eq!(nf[[0, 2]], 0.0,  "no formal charge");
    assert_eq!(nf[[0, 3]], 4.0,  "methane has 4 implicit H");
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
    // pyridine: c1ccncc1
    let mol = ingest("c1ccncc1").unwrap();
    let (nf, _, _) = mol_to_graph_arrays_raw(&mol);
    // 4th atom in parse order is N (atomic_num=7)
    assert_eq!(nf[[3, 0]], 7.0, "pyridine N atomic_num");
    assert_eq!(nf[[3, 1]], 1.0, "pyridine N is aromatic");
}

// ── edge index structure ────────────────────────────────────────────────────

#[test]
fn edge_index_bidirectional() {
    // ethane C-C: 1 bond → 2 directed edges (0→1) and (1→0)
    let mol = ingest("CC").unwrap();
    let (_, ei, _) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(ei.shape(), &[2, 2]);
    // One direction must be 0→1 and the other 1→0
    let pairs: Vec<(i64, i64)> = (0..2).map(|i| (ei[[0, i]], ei[[1, i]])).collect();
    assert!(pairs.contains(&(0, 1)), "must have 0→1 edge");
    assert!(pairs.contains(&(1, 0)), "must have 1→0 edge");
}

// ── edge feature one-hot encoding ──────────────────────────────────────────

#[test]
fn single_bond_one_hot() {
    // ethane C-C: single bond → [1,0,0,0]
    let mol = ingest("CC").unwrap();
    let (_, _, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(&ef.row(0).to_vec(), &[1.0_f32, 0.0, 0.0, 0.0]);
}

#[test]
fn double_bond_one_hot() {
    // ethylene C=C: double bond → [0,1,0,0]
    let mol = ingest("C=C").unwrap();
    let (_, _, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(&ef.row(0).to_vec(), &[0.0_f32, 1.0, 0.0, 0.0]);
}

#[test]
fn triple_bond_one_hot() {
    // acetylene C#C: triple bond → [0,0,1,0]
    let mol = ingest("C#C").unwrap();
    let (_, _, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(&ef.row(0).to_vec(), &[0.0_f32, 0.0, 1.0, 0.0]);
}

#[test]
fn aromatic_bond_one_hot() {
    // benzene: aromatic bond → [0,0,0,1]
    let mol = ingest("c1ccccc1").unwrap();
    let (_, _, ef) = mol_to_graph_arrays_raw(&mol);
    assert_eq!(&ef.row(0).to_vec(), &[0.0_f32, 0.0, 0.0, 1.0]);
}
