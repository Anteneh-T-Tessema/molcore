use molcore::ingest::ingest;

fn hs(smiles: &str) -> Vec<u8> {
    let mol = ingest(smiles).expect(smiles);
    mol.graph.node_indices().map(|i| mol.graph[i].num_hs).collect()
}

// ── aliphatic chains ────────────────────────────────────────────────────────

#[test]
fn methane_has_four_hs() {
    assert_eq!(hs("C"), vec![4]);
}

#[test]
fn ethanol_hs_correct() {
    // CCO: CH3(3H) - CH2(2H) - OH(1H)
    assert_eq!(hs("CCO"), vec![3, 2, 1]);
}

#[test]
fn acetic_acid_hs_correct() {
    // CC(=O)O: CH3(3) - C(0) = O(0) - OH(1)
    assert_eq!(hs("CC(=O)O"), vec![3, 0, 0, 1]);
}

// ── aromatic rings ──────────────────────────────────────────────────────────

#[test]
fn benzene_each_carbon_one_h() {
    let h = hs("c1ccccc1");
    assert_eq!(h.len(), 6);
    assert!(h.iter().all(|&v| v == 1), "all benzene C should have 1H, got {:?}", h);
}

#[test]
fn toluene_ipso_carbon_zero_h() {
    // Cc1ccccc1: methyl(3H), then ring C bonded to methyl(0H), 4×ring C(1H)
    let h = hs("Cc1ccccc1");
    assert_eq!(h[0], 3, "methyl CH3");
    assert_eq!(h[1], 0, "ipso ring carbon bonded to methyl");
    assert!(h[2..].iter().all(|&v| v == 1), "ortho/meta/para ring C should be 1H");
}

// ── heteroaromatics ─────────────────────────────────────────────────────────

#[test]
fn pyridine_nitrogen_zero_h() {
    // c1ccncc1: N has 2 aromatic bonds → valence 3 - (2×1.5 + 0.5 aromatic adj) = 0H
    let h = hs("c1ccncc1");
    let n_idx = 3; // 4th atom in parse order
    assert_eq!(h[n_idx], 0, "pyridine N should have 0H, got {:?}", h);
}

// ── functional groups ───────────────────────────────────────────────────────

#[test]
fn double_bond_carbon_zero_h() {
    // C=C: each alkene C has 2H (valence 4 - 1 single - 2 double = ... wait)
    // C=C: C connected by double bond: 4 - 2 = 2H each
    let h = hs("C=C");
    assert_eq!(h, vec![2, 2], "ethylene: each C has 2H");
}

#[test]
fn triple_bond_nitrogen_zero_h() {
    // C#N: nitrile — C has 1H (4 - 1 single - 3 triple... wait 4-1-3=-? no)
    // C≡N: C has valence 4, bonds = 1 (N triple = 3) → C: 4 - 3 = 1H; N: 3 - 3 = 0H
    let h = hs("C#N");
    assert_eq!(h[0], 1, "nitrile C should have 1H");
    assert_eq!(h[1], 0, "nitrile N should have 0H");
}

#[test]
fn amine_nitrogen_two_h() {
    // CN: methylamine, N has 1 bond to C → 3 - 1 = 2H
    let h = hs("CN");
    assert_eq!(h[1], 2, "methylamine N should have 2H");
}

#[test]
fn ether_oxygen_zero_h() {
    // COC: dimethyl ether, O has 2 bonds → 2 - 2 = 0H
    let h = hs("COC");
    assert_eq!(h[1], 0, "ether O should have 0H");
}

// ── benzoic acid full check ─────────────────────────────────────────────────

#[test]
fn benzoic_acid_hs() {
    // c1ccccc1C(=O)O
    // Ring: 4 CH (1H) + 1 ipso C (0H) + 1 meta C (1H)?
    // Actually: 5 ring C with 1H, 1 ipso C with 0H, carbonyl C (0H), =O (0H), OH (1H)
    let h = hs("c1ccccc1C(=O)O");
    let total: u32 = h.iter().map(|&v| v as u32).sum();
    assert_eq!(total, 6, "benzoic acid: 5 ring H + 1 OH = 6H total, got {:?}", h);
    // ipso C (idx 5, bonded to carboxyl) must be 0H
    assert_eq!(h[5], 0, "ipso C should have 0H");
    // carbonyl C (idx 6) = 0H
    assert_eq!(h[6], 0, "carbonyl C should have 0H");
    // =O (idx 7) = 0H
    assert_eq!(h[7], 0, "carbonyl O should have 0H");
    // -OH (idx 8) = 1H
    assert_eq!(h[8], 1, "-OH should have 1H");
}
