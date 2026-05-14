"""
Tests for agentic_rag — offline only (no network calls).
Tests the query rewriting, routing logic, and relevance scoring.
"""
import pytest
from molcore.agentic_rag import (
    rewrite_query,
    score_relevance,
    synthesize_answer,
    retrieve_local_tools,
    RetrievedChunk,
    ChemRAG,
)


def test_rewrite_expands_aspirin():
    rewritten, smiles = rewrite_query("What is the logP of aspirin?")
    assert "CC(=O)Oc1ccccc1C(=O)O" in smiles
    assert "aspirin" in rewritten


def test_rewrite_no_match_returns_original():
    q = "What is the boiling point of water?"
    rewritten, smiles = rewrite_query(q)
    assert smiles == []


def test_rewrite_multiple_drugs():
    _, smiles = rewrite_query("Compare aspirin and ibuprofen logP")
    assert len(smiles) == 2


def test_score_relevance_empty_chunks():
    assert score_relevance("What is the logP?", []) == 0.0


def test_score_relevance_relevant_content():
    chunks = [RetrievedChunk(source="test", content="XLogP=3.5 LogP logp", score=1.0)]
    score = score_relevance("What is the logP?", chunks)
    assert score >= 7.0, f"Expected high relevance, got {score}"


def test_score_relevance_irrelevant_content():
    chunks = [RetrievedChunk(source="test", content="completely unrelated text", score=1.0)]
    score = score_relevance("What is the logP?", chunks)
    assert score < 7.0


def test_synthesize_answer_empty():
    ans = synthesize_answer("query", [])
    assert "No information" in ans


def test_synthesize_answer_with_chunks():
    chunks = [RetrievedChunk(source="pubchem", content="MW=180.16, XLogP=-3.0")]
    ans = synthesize_answer("What is the MW?", chunks)
    assert "PUBCHEM" in ans
    assert "180" in ans


def test_local_tools_returns_chunks():
    chunks = retrieve_local_tools("descriptors query", ["CCO", "c1ccccc1"])
    assert len(chunks) == 2
    assert all(c.source == "local_tools" for c in chunks)
    assert all("MW=" in c.content for c in chunks)


def test_local_tools_empty_smiles():
    chunks = retrieve_local_tools("no smiles query", [])
    assert chunks == []


def test_rag_query_offline_with_known_smiles():
    """RAG with explicit SMILES in query — local_tools path, no network needed."""
    rag = ChemRAG()
    # Put SMILES directly in query so PubChem lookup is minimal
    result = rag.query("Properties of SMILES CCO ethanol (SMILES: CCO)", verbose=False)
    assert result.iterations >= 1
    assert "local_tools" in result.sources
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


def test_rag_max_iterations_reached():
    """A nonsense query should exhaust iterations without crashing."""
    rag = ChemRAG()
    result = rag.query("zzzzzzzzz meaningless query with no chemistry")
    assert result.iterations <= 5


def test_rag_result_dataclass_fields():
    rag = ChemRAG()
    result = rag.query("aspirin logP")
    assert hasattr(result, "answer")
    assert hasattr(result, "sources")
    assert hasattr(result, "iterations")
    assert hasattr(result, "relevant")
