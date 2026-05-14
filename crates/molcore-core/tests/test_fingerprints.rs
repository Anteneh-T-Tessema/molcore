use molcore::fingerprints::compute_morgan;

#[test]
fn fingerprint_is_deterministic() {
    let fp1 = compute_morgan("c1ccccc1", 2, 2048);
    let fp2 = compute_morgan("c1ccccc1", 2, 2048);
    assert_eq!(fp1, fp2, "fingerprints must be deterministic");
}

#[test]
fn fingerprint_correct_length() {
    assert_eq!(compute_morgan("CCO", 2, 2048).len(), 2048);
    assert_eq!(compute_morgan("CCO", 2, 1024).len(), 1024);
}

#[test]
fn fingerprint_is_binary() {
    let fp = compute_morgan("c1ccccc1", 2, 2048);
    for &b in &fp {
        assert!(b == 0 || b == 1, "all bits must be 0 or 1");
    }
}

#[test]
fn different_molecules_different_fingerprints() {
    let fp1 = compute_morgan("CCO", 2, 2048);
    let fp2 = compute_morgan("c1ccccc1", 2, 2048);
    assert_ne!(fp1, fp2, "ethanol and benzene fingerprints must differ");
}

#[test]
fn same_molecule_same_fingerprint_regardless_of_radius() {
    // Same mol, different radius → different fps (more bits set at higher radius)
    let fp2 = compute_morgan("c1ccccc1", 2, 2048);
    let fp3 = compute_morgan("c1ccccc1", 3, 2048);
    // radius 3 sets ≥ as many bits as radius 2 (superset of environment shells)
    let bits2: u32 = fp2.iter().map(|&b| b as u32).sum();
    let bits3: u32 = fp3.iter().map(|&b| b as u32).sum();
    assert!(bits3 >= bits2, "larger radius should set at least as many bits");
}

#[test]
fn invalid_smiles_returns_zero_fingerprint() {
    // Graceful fallback: zeros, not a panic
    let fp = compute_morgan("INVALID_SMILES_XYZ", 2, 2048);
    assert_eq!(fp.len(), 2048);
    assert!(fp.iter().all(|&b| b == 0), "invalid SMILES → zero fingerprint");
}

#[test]
fn fingerprint_has_at_least_one_bit_set() {
    let fp = compute_morgan("C", 2, 2048); // methane
    let bits: u32 = fp.iter().map(|&b| b as u32).sum();
    assert!(bits > 0, "fingerprint of methane must have at least one bit set");
}
