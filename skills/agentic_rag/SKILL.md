# Skill: agentic_rag

## What this is
Iterative chemical literature retrieval with query rewriting and multi-source routing,
matching the Agentic RAG pattern (see architecture diagrams).

## Flow
```
User query (chemistry)
  1. Rewrite Query     → expand acronyms, normalize SMILES, add synonyms
  2. Updated Query     → needs more details? YES → choose source
  3. Source routing    → Vector DB | tools/local | Internet (PubChem/ChEMBL)
  4. Retrieved context → LLM generates response
  5. Relevance check  → relevant? YES → Final Response | NO → iterate (max 5)
```

## When to invoke
- "What is the mechanism of action of [drug]?"
- "Find similar compounds to [SMILES] with IC50 < 100nM"
- "What assays have been run on [compound name]?"

## Sources used
| Source | When | Tool |
|---|---|---|
| Vector DB | Embedded literature, patents | `tools/local/similarity_search` |
| PubChem API | Structure lookup, synonyms, properties | `tools/mcp/pubchem_mcp.yaml` |
| ChEMBL API | Bioactivity data | `tools/mcp/chembl_mcp.yaml` |
| Internet | Novel compounds, preprints | web search |

## Guardrails
- Max 5 retrieval iterations — never infinite loop
- Relevance gate: LLM self-scores answer 1–10; iterate if < 7
- All SMILES extracted from retrieved text are validated before use
