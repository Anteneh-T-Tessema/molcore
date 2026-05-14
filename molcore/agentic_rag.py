"""
molcore.agentic_rag — iterative chemical literature retrieval.

Implements the Agentic RAG pattern from skills/agentic_rag/SKILL.md:
  1. Rewrite query (normalize SMILES, expand acronyms)
  2. Route to sources (Vector DB | local tools | PubChem | ChEMBL)
  3. Retrieve context
  4. Check relevance (score 1–10) — iterate if < RELEVANCE_THRESHOLD
  5. Return Final Response

Usage:
    from molcore.agentic_rag import ChemRAG
    rag = ChemRAG()
    result = rag.query("What is the logP of aspirin?")
    print(result.answer, result.sources, result.iterations)
"""
from __future__ import annotations

import re
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

MAX_ITERATIONS      = 5
RELEVANCE_THRESHOLD = 7      # score 1–10; iterate if below
PUBCHEM_BASE        = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE         = "https://www.ebi.ac.uk/chembl/api/data"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    source:  str
    content: str
    score:   float = 1.0


@dataclass
class RAGResult:
    answer:      str
    sources:     list[str]
    iterations:  int
    relevant:    bool
    raw_chunks:  list[RetrievedChunk] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

# Common drug/molecule name → SMILES or canonical name
_SYNONYMS = {
    "aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":  "CC(=O)Nc1ccc(O)cc1",
    "caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "ethanol":      "CCO",
    "benzene":      "c1ccccc1",
    "acetone":      "CC(=O)C",
}

_SMILES_RE = re.compile(r'\b([A-Za-z0-9@+\-\[\]\(\)=#%/\\\.]+)\b')


def rewrite_query(query: str) -> tuple[str, list[str]]:
    """
    Rewrite a chemistry query:
    - Expand known drug names to SMILES
    - Identify embedded SMILES strings
    Returns (rewritten_query, extracted_smiles).
    """
    q = query.lower()
    extracted: list[str] = []

    for name, smi in _SYNONYMS.items():
        if name in q:
            query = query.replace(name, f"{name} (SMILES: {smi})")
            extracted.append(smi)

    return query, extracted


# ---------------------------------------------------------------------------
# Source routing + retrieval
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def retrieve_pubchem(query: str, smiles: list[str]) -> list[RetrievedChunk]:
    """Look up properties from PubChem for any SMILES found in the query."""
    chunks = []
    for smi in smiles:
        encoded = urllib.parse.quote(smi)
        url = f"{PUBCHEM_BASE}/compound/smiles/{encoded}/property/MolecularFormula,MolecularWeight,XLogP,TPSA/JSON"
        data = _fetch_json(url)
        if data and "PropertyTable" in data:
            props = data["PropertyTable"]["Properties"][0]
            content = (
                f"PubChem properties for SMILES={smi}: "
                f"Formula={props.get('MolecularFormula','?')}, "
                f"MW={props.get('MolecularWeight','?')}, "
                f"XLogP={props.get('XLogP','?')}, "
                f"TPSA={props.get('TPSA','?')}"
            )
            chunks.append(RetrievedChunk(source="pubchem", content=content, score=0.9))
    return chunks


def retrieve_pubchem_by_name(query: str) -> list[RetrievedChunk]:
    """Look up by compound name when no SMILES is available."""
    words = [w for w in query.split() if len(w) > 4 and w.isalpha()]
    chunks = []
    for word in words[:3]:
        encoded = urllib.parse.quote(word)
        url = f"{PUBCHEM_BASE}/compound/name/{encoded}/property/IsomericSMILES,MolecularWeight,XLogP/JSON"
        data = _fetch_json(url)
        if data and "PropertyTable" in data:
            props = data["PropertyTable"]["Properties"][0]
            content = (
                f"PubChem: '{word}' → "
                f"SMILES={props.get('IsomericSMILES','?')}, "
                f"MW={props.get('MolecularWeight','?')}, "
                f"XLogP={props.get('XLogP','?')}"
            )
            chunks.append(RetrievedChunk(source="pubchem_name", content=content, score=0.85))
    return chunks


def retrieve_local_tools(query: str, smiles: list[str]) -> list[RetrievedChunk]:
    """Use local molcore tools for structural queries."""
    chunks = []
    if smiles:
        from molcore.featurizers.descriptors import calc_descriptors
        desc = calc_descriptors(smiles, backend="rdkit")
        for smi, row in zip(smiles, desc.tolist()):
            content = (
                f"Local descriptors for {smi}: "
                f"MW={row[0]:.2f}, LogP={row[1]:.2f}, TPSA={row[2]:.2f}"
            )
            chunks.append(RetrievedChunk(source="local_tools", content=content, score=1.0))
    return chunks


def route_and_retrieve(query: str, smiles: list[str], iteration: int) -> list[RetrievedChunk]:
    """Route query to appropriate sources based on content and iteration."""
    chunks: list[RetrievedChunk] = []

    # Always try local tools first if SMILES available (fast, no network)
    chunks.extend(retrieve_local_tools(query, smiles))

    # PubChem for structure-based lookup
    if smiles:
        chunks.extend(retrieve_pubchem(query, smiles))
    else:
        chunks.extend(retrieve_pubchem_by_name(query))

    # On retry iterations, broaden to name-based search if no SMILES
    if iteration > 1 and not smiles:
        chunks.extend(retrieve_pubchem_by_name(query))

    return chunks


# ---------------------------------------------------------------------------
# Relevance scoring (rule-based — no LLM required)
# ---------------------------------------------------------------------------

_PROPERTY_KEYWORDS = {
    "logp": ["xlogp", "logp", "log p", "lipophilicity"],
    "mw":   ["molecular weight", "mw", "mass", "dalton"],
    "tpsa": ["tpsa", "polar surface area"],
    "smiles": ["smiles", "structure"],
    "formula": ["formula", "molecular formula"],
}


def score_relevance(query: str, chunks: list[RetrievedChunk]) -> float:
    """
    Score how relevant the retrieved chunks are to the query (1–10).
    Rule-based: checks keyword overlap between query intent and retrieved content.
    """
    if not chunks:
        return 0.0
    query_lower = query.lower()

    # Determine what property is being asked about
    asked_props = set()
    for prop, keywords in _PROPERTY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            asked_props.add(prop)

    if not asked_props:
        asked_props = {"any"}

    combined = " ".join(c.content.lower() for c in chunks)
    matches = 0
    for prop in asked_props:
        if prop == "any":
            matches += 1
        else:
            if any(kw in combined for kw in _PROPERTY_KEYWORDS.get(prop, [])):
                matches += 1

    base_score = (matches / max(len(asked_props), 1)) * 8
    has_numbers = bool(re.search(r'\d+\.?\d*', combined))
    return min(10.0, base_score + (2.0 if has_numbers else 0.0))


# ---------------------------------------------------------------------------
# Answer synthesis (rule-based)
# ---------------------------------------------------------------------------

def synthesize_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return f"No information found for: {query}"
    parts = [f"Based on {len(chunks)} source(s):"]
    for chunk in chunks:
        parts.append(f"  [{chunk.source.upper()}] {chunk.content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main RAG orchestrator
# ---------------------------------------------------------------------------

class ChemRAG:
    """
    Agentic RAG for chemistry queries.

    For LLM-powered query rewriting and answer synthesis, subclass this and
    override `rewrite`, `check_relevance`, and `synthesize`.
    """

    def query(self, user_query: str, verbose: bool = False) -> RAGResult:
        sources_used: list[str] = []
        all_chunks:   list[RetrievedChunk] = []

        current_query, smiles = rewrite_query(user_query)
        if verbose:
            print(f"[RAG] Query: {current_query[:80]}")
            if smiles:
                print(f"[RAG] Extracted SMILES: {smiles}")

        for iteration in range(1, MAX_ITERATIONS + 1):
            chunks = route_and_retrieve(current_query, smiles, iteration)
            all_chunks.extend(chunks)
            sources_used.extend(c.source for c in chunks)

            score = score_relevance(current_query, chunks)
            if verbose:
                print(f"[RAG] Iteration {iteration}: {len(chunks)} chunks, relevance={score:.1f}/10")

            if score >= RELEVANCE_THRESHOLD:
                answer = synthesize_answer(user_query, chunks)
                return RAGResult(
                    answer=answer,
                    sources=list(dict.fromkeys(sources_used)),
                    iterations=iteration,
                    relevant=True,
                    raw_chunks=all_chunks,
                )

            # Refine: broaden query on retry
            current_query = f"{user_query} properties structure synonyms"

        # Max iterations reached — return best available
        answer = synthesize_answer(user_query, all_chunks) if all_chunks else (
            f"Could not find sufficient information for: {user_query}"
        )
        return RAGResult(
            answer=answer,
            sources=list(dict.fromkeys(sources_used)),
            iterations=MAX_ITERATIONS,
            relevant=False,
            raw_chunks=all_chunks,
        )
