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
    BindingRecord,
    _parse_chembl_record,
    _parse_zinc_record,
    chembl_search,
    chembl_by_id,
    chembl_activity,
    chembl_smiles,
    zinc_by_id,
    zinc_smiles,
    zinc_subsets,
    zinc_random_sample,
    tdc_dataset,
    bindingdb_search,
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


# ── tdc_dataset mocked ───────────────────────────────────────────────────────

def _make_tdc_split():
    import pandas as pd
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1"], "Y": [1.0, 0.0]})
    return {"train": df, "valid": df.head(1), "test": df.head(1)}


def test_tdc_dataset_returns_splits(monkeypatch):
    splits = _make_tdc_split()
    mock_data = MagicMock()
    mock_data.get_split.return_value = splits
    mock_admet = MagicMock(return_value=mock_data)

    import types
    fake_tdc = types.ModuleType("tdc")
    fake_single = types.ModuleType("tdc.single_pred")
    fake_single.ADMET = mock_admet
    fake_tdc.single_pred = fake_single

    monkeypatch.setitem(__import__("sys").modules, "tdc", fake_tdc)
    monkeypatch.setitem(__import__("sys").modules, "tdc.single_pred", fake_single)

    result = tdc_dataset("BBB_Martini")
    assert set(result.keys()) == {"train", "valid", "test"}
    assert len(result["train"]) == 2


def test_tdc_dataset_missing_dep_raises(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "tdc", None)
    with pytest.raises(ImportError, match="PyTDC"):
        tdc_dataset("BBB_Martini")


def test_tdc_dataset_unknown_name_raises(monkeypatch):
    import types, sys
    fake_tdc = types.ModuleType("tdc")
    fake_single = types.ModuleType("tdc.single_pred")

    def _raise(*a, **kw):
        raise ValueError("unknown dataset")

    fake_single.ADMET = _raise
    fake_tdc.single_pred = fake_single
    fake_multi = types.ModuleType("tdc.multi_pred")

    def _raise2(*a, **kw):
        raise ValueError("unknown dataset")

    fake_multi.DTI = _raise2
    fake_tdc.multi_pred = fake_multi

    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single)
    monkeypatch.setitem(sys.modules, "tdc.multi_pred", fake_multi)

    with pytest.raises(ValueError, match="Could not load TDC dataset"):
        tdc_dataset("NONEXISTENT_DATASET_XYZ")


# ── bindingdb_search mocked ──────────────────────────────────────────────────

def _make_bindingdb_split():
    import pandas as pd
    # No Target_ID column so the filter uses the Target column (protein names)
    df = pd.DataFrame({
        "Drug":   ["CCO", "c1ccccc1", "CC(=O)O"],
        "Target": ["EGFR_HUMAN", "EGFR_HUMAN", "CDK2_HUMAN"],
        "Y":      [8.5,          7.2,          6.1],
    })
    return {"train": df, "valid": df.head(1), "test": df.head(1)}


def test_bindingdb_search_returns_records(monkeypatch):
    monkeypatch.setattr("molcore.databases.tdc_dataset", lambda *a, **kw: _make_bindingdb_split())
    records = bindingdb_search(affinity="Kd")
    assert len(records) == 3
    assert all(isinstance(r, BindingRecord) for r in records)


def test_bindingdb_search_target_filter(monkeypatch):
    monkeypatch.setattr("molcore.databases.tdc_dataset", lambda *a, **kw: _make_bindingdb_split())
    records = bindingdb_search(affinity="Kd", target="EGFR")
    assert len(records) == 2
    assert all("EGFR" in r.target_sequence for r in records)


def test_bindingdb_search_max_records(monkeypatch):
    monkeypatch.setattr("molcore.databases.tdc_dataset", lambda *a, **kw: _make_bindingdb_split())
    records = bindingdb_search(affinity="Kd", max_records=1)
    assert len(records) == 1


def test_bindingdb_search_affinity_type_stored(monkeypatch):
    monkeypatch.setattr("molcore.databases.tdc_dataset", lambda *a, **kw: _make_bindingdb_split())
    records = bindingdb_search(affinity="IC50")
    assert all(r.affinity_type == "IC50" for r in records)


def test_bindingdb_record_fields(monkeypatch):
    monkeypatch.setattr("molcore.databases.tdc_dataset", lambda *a, **kw: _make_bindingdb_split())
    r = bindingdb_search(affinity="Kd")[0]
    assert r.smiles == "CCO"
    assert r.affinity == pytest.approx(8.5)
    assert r.affinity_type == "Kd"


# ── chembl_activity mocked ───────────────────────────────────────────────────

def test_chembl_activity_returns_list(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: {"activities": [
            {"molecule_chembl_id": "CHEMBL25", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
             "standard_value": "150.0", "standard_units": "nM", "assay_chembl_id": "CHEMBL123"},
        ]},
    )
    results = chembl_activity("CHEMBL240")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["molecule_chembl_id"] == "CHEMBL25"


def test_chembl_activity_empty_on_no_data(monkeypatch):
    monkeypatch.setattr("molcore.databases._get_json", lambda *a, **kw: {})
    results = chembl_activity("CHEMBL_NONE")
    assert results == []


# ── chembl_smiles mocked ─────────────────────────────────────────────────────

def test_chembl_smiles_batch_returns_dict(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: {"molecules": [
            {"molecule_chembl_id": "CHEMBL25",
             "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
            {"molecule_chembl_id": "CHEMBL192",
             "molecule_structures": {"canonical_smiles": "CCO"}},
        ]},
    )
    monkeypatch.setattr("molcore.databases.time.sleep", lambda _: None)
    result = chembl_smiles(["CHEMBL25", "CHEMBL192"])
    assert result == {"CHEMBL25": "CC(=O)Oc1ccccc1C(=O)O", "CHEMBL192": "CCO"}


def test_chembl_smiles_skips_no_structure(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: {"molecules": [
            {"molecule_chembl_id": "CHEMBL999", "molecule_structures": None},
        ]},
    )
    monkeypatch.setattr("molcore.databases.time.sleep", lambda _: None)
    result = chembl_smiles(["CHEMBL999"])
    assert result == {}


# ── zinc_subsets ─────────────────────────────────────────────────────────────

def test_zinc_subsets_returns_list(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: [{"name": "Drug-Like"}, {"name": "Lead-Like"}],
    )
    subsets = zinc_subsets()
    assert isinstance(subsets, list)
    assert "Drug-Like" in subsets


def test_zinc_subsets_fallback_on_error(monkeypatch):
    monkeypatch.setattr("molcore.databases._get_json", lambda *a, **kw: (_ for _ in ()).throw(Exception("net err")))
    subsets = zinc_subsets()
    assert isinstance(subsets, list)
    assert len(subsets) > 0


# ── zinc_random_sample ────────────────────────────────────────────────────────

def test_zinc_random_sample_returns_list(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases._get_json",
        lambda *a, **kw: [{"smiles": "CCO"}, {"smiles": "c1ccccc1"}],
    )
    result = zinc_random_sample(n=2)
    assert isinstance(result, list)
    assert result == ["CCO", "c1ccccc1"]


def test_zinc_random_sample_empty_on_error(monkeypatch):
    monkeypatch.setattr("molcore.databases._get_json", lambda *a, **kw: (_ for _ in ()).throw(Exception("net err")))
    result = zinc_random_sample(n=10)
    assert result == []
