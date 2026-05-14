use petgraph::stable_graph::NodeIndex;
use std::collections::HashMap;
use thiserror::Error;
use crate::molecule::{Atom, Bond, BondType, MolGraph, PyMolGraph};

#[derive(Debug, Error)]
pub enum IngestionError {
    #[error("invalid SMILES: {0}")]
    InvalidSmiles(String),
}

/// Ingestion pipeline:
///   SMILES string → parse → petgraph StableGraph → PyMolGraph (immutable)
///
/// When `rdkit-backend` feature is enabled, rdkit-rs handles sanitization,
/// aromaticity perception, and canonical SMILES. Otherwise a built-in
/// recursive-descent parser handles common SMILES for bootstrap/testing.
pub fn ingest(smiles: &str) -> Result<PyMolGraph, IngestionError> {
    #[cfg(feature = "rdkit-backend")]
    return ingest_rdkit(smiles);
    #[cfg(not(feature = "rdkit-backend"))]
    return ingest_builtin(smiles);
}

// ---------------------------------------------------------------------------
// Built-in SMILES parser (no external crate dependency)
// Handles: aliphatic/aromatic atoms, branches, ring closures, bracket atoms,
// explicit bonds. Sufficient for all test cases and common drug-like SMILES.
// ---------------------------------------------------------------------------

fn push_atom(
    graph: &mut MolGraph,
    last: &mut Option<NodeIndex>,
    next_bond: &mut Option<BondType>,
    atomic_num: u8,
    is_aromatic: bool,
) {
    let idx = graph.add_node(Atom { atomic_num, is_aromatic, formal_charge: 0, num_hs: 0 });
    if let Some(prev) = *last {
        let default = if is_aromatic { BondType::Aromatic } else { BondType::Single };
        let bt = next_bond.take().unwrap_or(default);
        graph.add_edge(prev, idx, Bond { bond_type: bt, is_aromatic: bt == BondType::Aromatic });
    }
    *last = Some(idx);
}

fn close_ring(
    graph: &mut MolGraph,
    ring_map: &mut HashMap<u8, NodeIndex>,
    current: Option<NodeIndex>,
    next_bond: &mut Option<BondType>,
    ring_num: u8,
) {
    if let Some(ring_atom) = ring_map.remove(&ring_num) {
        if let Some(cur) = current {
            let bt = next_bond.take().unwrap_or(BondType::Single);
            graph.add_edge(ring_atom, cur, Bond { bond_type: bt, is_aromatic: bt == BondType::Aromatic });
        }
    } else if let Some(cur) = current {
        ring_map.insert(ring_num, cur);
        *next_bond = None;
    }
}

fn bracket_atomic_num(inner: &str) -> (u8, bool) {
    let s = inner.trim_start_matches(|c: char| c.is_ascii_digit());
    let sym: String = s.chars().take_while(|c| c.is_alphabetic()).collect();
    let is_ar = sym.chars().next().map(|c| c.is_lowercase()).unwrap_or(false);
    let n = match sym.to_lowercase().as_str() {
        "h"  =>  1, "he" =>  2, "li" =>  3, "be" =>  4, "b"  =>  5, "c"  =>  6,
        "n"  =>  7, "o"  =>  8, "f"  =>  9, "ne" => 10, "na" => 11, "mg" => 12,
        "al" => 13, "si" => 14, "p"  => 15, "s"  => 16, "cl" => 17, "ar" => 18,
        "k"  => 19, "ca" => 20, "fe" => 26, "co" => 27, "ni" => 28, "cu" => 29,
        "zn" => 30, "br" => 35, "kr" => 36, "ag" => 47, "i"  => 53, "xe" => 54,
        _ => 0,
    };
    (n, is_ar)
}

fn ingest_builtin(smiles: &str) -> Result<PyMolGraph, IngestionError> {
    let chars: Vec<char> = smiles.chars().collect();
    let len = chars.len();
    let mut graph = MolGraph::new();
    let mut i = 0usize;
    let mut last: Option<NodeIndex> = None;
    let mut branch_stack: Vec<Option<NodeIndex>> = Vec::new();
    let mut ring_map: HashMap<u8, NodeIndex> = HashMap::new();
    let mut next_bond: Option<BondType> = None;

    while i < len {
        let c = chars[i];
        match c {
            // Aromatic atoms
            'b' => { push_atom(&mut graph, &mut last, &mut next_bond, 5,  true);  i += 1; }
            'c' => { push_atom(&mut graph, &mut last, &mut next_bond, 6,  true);  i += 1; }
            'n' => { push_atom(&mut graph, &mut last, &mut next_bond, 7,  true);  i += 1; }
            'o' => { push_atom(&mut graph, &mut last, &mut next_bond, 8,  true);  i += 1; }
            'p' => { push_atom(&mut graph, &mut last, &mut next_bond, 15, true);  i += 1; }
            's' => { push_atom(&mut graph, &mut last, &mut next_bond, 16, true);  i += 1; }
            // Aliphatic atoms (multi-char: Cl, Br)
            'B' => {
                if i + 1 < len && chars[i+1] == 'r' {
                    push_atom(&mut graph, &mut last, &mut next_bond, 35, false); i += 2;
                } else {
                    push_atom(&mut graph, &mut last, &mut next_bond, 5,  false); i += 1;
                }
            }
            'C' => {
                if i + 1 < len && chars[i+1] == 'l' {
                    push_atom(&mut graph, &mut last, &mut next_bond, 17, false); i += 2;
                } else {
                    push_atom(&mut graph, &mut last, &mut next_bond, 6,  false); i += 1;
                }
            }
            'N' => { push_atom(&mut graph, &mut last, &mut next_bond, 7,  false); i += 1; }
            'O' => { push_atom(&mut graph, &mut last, &mut next_bond, 8,  false); i += 1; }
            'F' => { push_atom(&mut graph, &mut last, &mut next_bond, 9,  false); i += 1; }
            'P' => { push_atom(&mut graph, &mut last, &mut next_bond, 15, false); i += 1; }
            'S' => { push_atom(&mut graph, &mut last, &mut next_bond, 16, false); i += 1; }
            'I' => { push_atom(&mut graph, &mut last, &mut next_bond, 53, false); i += 1; }
            // Explicit bonds
            '=' => { next_bond = Some(BondType::Double);   i += 1; }
            '#' => { next_bond = Some(BondType::Triple);   i += 1; }
            ':' => { next_bond = Some(BondType::Aromatic); i += 1; }
            '-' => { next_bond = Some(BondType::Single);   i += 1; }
            // Branches
            '(' => { branch_stack.push(last); i += 1; }
            ')' => {
                last = branch_stack.pop()
                    .ok_or_else(|| IngestionError::InvalidSmiles("unmatched ')'".into()))?;
                i += 1;
            }
            // Single-digit ring closure
            '0'..='9' => {
                let n = c as u8 - b'0';
                close_ring(&mut graph, &mut ring_map, last, &mut next_bond, n);
                i += 1;
            }
            // Two-digit ring closure: %10, %11, ...
            '%' => {
                if i + 2 < len && chars[i+1].is_ascii_digit() && chars[i+2].is_ascii_digit() {
                    let n = (chars[i+1] as u8 - b'0') * 10 + (chars[i+2] as u8 - b'0');
                    close_ring(&mut graph, &mut ring_map, last, &mut next_bond, n);
                    i += 3;
                } else {
                    return Err(IngestionError::InvalidSmiles(format!("bad '%' at pos {i}")));
                }
            }
            // Bracket atoms: [Na+], [NH2], [2H], etc.
            '[' => {
                let rel_end = chars[i..].iter().position(|&ch| ch == ']')
                    .ok_or_else(|| IngestionError::InvalidSmiles("unclosed '['".into()))?;
                let inner: String = chars[i+1..i+rel_end].iter().collect();
                let (anum, is_ar) = bracket_atomic_num(&inner);
                push_atom(&mut graph, &mut last, &mut next_bond, anum, is_ar);
                i += rel_end + 1;
            }
            // Skip: stereochemistry, disconnected component separator, whitespace
            '/' | '\\' | '@' | '.' | ' ' | '\t' => { i += 1; }
            _ => return Err(IngestionError::InvalidSmiles(
                format!("unexpected character '{c}' at position {i}")
            )),
        }
    }

    if graph.node_count() == 0 {
        return Err(IngestionError::InvalidSmiles("no atoms parsed".into()));
    }
    if !ring_map.is_empty() {
        return Err(IngestionError::InvalidSmiles(
            format!("unclosed ring bond(s): {:?}", ring_map.keys().collect::<Vec<_>>())
        ));
    }

    Ok(PyMolGraph { graph, canonical_smiles: smiles.to_string() })
}

#[cfg(feature = "rdkit-backend")]
fn ingest_rdkit(_smiles: &str) -> Result<PyMolGraph, IngestionError> {
    todo!("rdkit-backend: wire rdkit-rs once linked in CI")
}
