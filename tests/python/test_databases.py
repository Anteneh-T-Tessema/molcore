"""
Tests for molcore.databases — offline unit tests only.

All network calls are stubbed; no internet connection required.
"""
from unittest.mock import patch, MagicMock
import json
import pytest

from molcore.databases import (
    ChEMBLCompound,
    ZINCCompound,
    _parse_chembl_record,
    _parse_zinc_record,
    chembl_search,
    chembl_by_id,
    zinc_by_id,
    zinc_smiles,
)


# ── ChEMBL record parsing ────────────────────────────────────────────────────

def test_parse_chembl_record_full():
    raw = {
        "molecule_chembl_id": "CHEMBL25",
        "pref_name": "ASPIRIN",
        "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        "molecule_properties": {
            "full_mwt": "180.16",
            "alogp": "1.31",
            "hba": "3",
            "hbd": "1",
            "psa": "63.60",
            "num_ro5_violations": "0",
        },
    }
    c = _parse_chembl_record(raw)
    assert c.chembl_id == "CHEMBL25"
    assert c.smiles    == "CC(=O)Oc1ccccc1C(=O)O"
    assert c.name      == "ASPIRIN"
    assert abs(c.mw - 180.16) < 0.01
    assert c.ro5_violations == 0


def test_parse_chembl_record_missing_structures():
    raw = {"molecule_chembl_id": "CHEMBL999", "pref_name": None,
           "molecule_structures": None, "molecule_properties": None}
    c = _parse_chembl_record(raw)
    assert c.smiles == ""
    assert c.name   == ""
    assert c.mw     == 0.0


def test_chembl_compound_dataclass():
    c = ChEMBLCompound(chembl_id="CHEMBL1", smiles="CCO")
    assert c.chembl_id == "CHEMBL1"
    assert c.smiles    == "CCO"


# ── ZINC record parsing ──────────────────────────────────────────────────────

def test_parse_zinc_record():
    raw = {"zinc_id": "ZINC000001234567", "smiles": "CCO", "mwt": "46.07", "logp": "-0.31"}
    z = _parse_zinc_record(raw)
    assert z.zinc_id == "ZINC000001234567"
    assert z.smiles  == "CCO"
    assert abs(z.mw - 46.07) < 0.01


def test_parse_zinc_record_empty():
    z = _parse_zinc_record({})
    assert z.zinc_id == ""
    assert z.smiles  == ""
    assert z.mw      == 0.0


def test_zinc_compound_dataclass():
    z = ZINCCompound(zinc_id="ZINC1", smiles="c1ccccc1")
    assert z.zinc_id == "ZINC1"


# ── chembl_search mocked ─────────────────────────────────────────────────────

def _mock_chembl_response(smiles="CC(=O)Oc1ccccc1C(=O)O"):
    return {
        "molecules": [{
            "molecule_chembl_id": "CHEMBL25",
            "pref_name": "ASPIRIN",
            "molecule_structures": {"canonical_smiles": smiles},
            "molecule_properties": {"full_mwt": "180.16", "alogp": "1.31",
                                    "hba": "3", "hbd": "1", "psa": "63.6",
                                    "num_ro5_violations": "0"},
        }]
    }


def test_chembl_search_returns_compounds(monkeypatch):
    monkeypatch.setattr("molcore.databases._get_json", lambda *a, **kw: _mock_chembl_response())
    results = chembl_search("aspirin")
    assert len(results) == 1
    assert results[0].chembl_id == "CHEMBL25"
    assert results[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"


def test_chembl_by_id_returns_compound(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: _mock_chembl_response()["molecules"][0],
    )
    c = chembl_by_id("CHEMBL25")
    assert c is not None
    assert c.chembl_id == "CHEMBL25"


def test_chembl_by_id_http_error_returns_none(monkeypatch):
    import urllib.error
    def raise_http(*a, **kw):
        raise urllib.error.HTTPError(None, 404, "Not Found", {}, None)
    monkeypatch.setattr("molcore.databases._get_json", raise_http)
    assert chembl_by_id("CHEMBL_NONEXISTENT") is None


# ── zinc_by_id mocked ────────────────────────────────────────────────────────

def test_zinc_by_id_returns_compound(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: {"zinc_id": "ZINC000001234567", "smiles": "CCO", "mwt": "46.07", "logp": "-0.31"},
    )
    z = zinc_by_id("ZINC000001234567")
    assert z is not None
    assert z.smiles == "CCO"


def test_zinc_smiles_batch(monkeypatch):
    calls = iter([
        {"zinc_id": "ZINC1", "smiles": "CCO",      "mwt": "46", "logp": "-0.3"},
        {"zinc_id": "ZINC2", "smiles": "c1ccccc1",  "mwt": "78", "logp": "1.6"},
    ])
    monkeypatch.setattr("molcore.databases._get_json", lambda *a, **kw: next(calls))
    monkeypatch.setattr("molcore.databases.time.sleep", lambda _: None)
    result = zinc_smiles(["ZINC1", "ZINC2"])
    assert result == {"ZINC1": "CCO", "ZINC2": "c1ccccc1"}
