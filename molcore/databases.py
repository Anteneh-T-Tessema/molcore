"""
molcore.databases — ChEMBL and ZINC connectors.

ChEMBL : REST API v3, no API key required.
ZINC   : ZINC20 Cartridge-free downloads (tranches or SMILES by ID).

All functions return plain Python lists so they compose naturally with
MolDataset.from_smiles() and featurize_smiles().
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "molcore/0.1 (cheminformatics; github.com/Anteneh-T-Tessema/molcore)"}


def _get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


# ---------------------------------------------------------------------------
# ChEMBL
# ---------------------------------------------------------------------------

_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass
class ChEMBLCompound:
    chembl_id: str
    smiles: str
    name: str = ""
    mw: float = 0.0
    alogp: float = 0.0
    hba: int = 0
    hbd: int = 0
    psa: float = 0.0
    ro5_violations: int = 0


def chembl_search(
    query: str,
    limit: int = 25,
    timeout: int = 15,
) -> list[ChEMBLCompound]:
    """
    Full-text search ChEMBL by name, synonym, or InChIKey.

    Returns up to `limit` compounds with SMILES and Lipinski properties.

    >>> results = chembl_search("aspirin", limit=5)
    >>> results[0].smiles
    'CC(=O)Oc1ccccc1C(=O)O'
    """
    encoded = urllib.parse.quote(query)
    url = (
        f"{_CHEMBL_BASE}/molecule.json"
        f"?pref_name__icontains={encoded}"
        f"&limit={limit}"
        f"&only=molecule_chembl_id,pref_name,molecule_structures,molecule_properties"
    )
    data = _get_json(url, timeout=timeout)
    return [_parse_chembl_record(m) for m in data.get("molecules", [])]


def chembl_by_id(chembl_id: str, timeout: int = 15) -> ChEMBLCompound | None:
    """Fetch a single compound by ChEMBL ID, e.g. 'CHEMBL25'."""
    url = f"{_CHEMBL_BASE}/molecule/{chembl_id}.json"
    try:
        data = _get_json(url, timeout=timeout)
    except urllib.error.HTTPError:
        return None
    return _parse_chembl_record(data)


def chembl_activity(
    chembl_id: str,
    standard_type: str = "IC50",
    limit: int = 100,
    timeout: int = 15,
) -> list[dict]:
    """
    Return bioactivity records for a target ChEMBL ID.

    standard_type: 'IC50', 'Ki', 'Kd', 'EC50', etc.
    Each record has keys: molecule_chembl_id, canonical_smiles, standard_value,
    standard_units, assay_chembl_id.
    """
    url = (
        f"{_CHEMBL_BASE}/activity.json"
        f"?target_chembl_id={chembl_id}"
        f"&standard_type={standard_type}"
        f"&limit={limit}"
    )
    data = _get_json(url, timeout=timeout)
    return data.get("activities", [])


def chembl_smiles(
    chembl_ids: list[str],
    batch_size: int = 50,
    timeout: int = 15,
    sleep_between: float = 0.3,
) -> dict[str, str]:
    """
    Batch-fetch canonical SMILES for a list of ChEMBL IDs.

    Returns {chembl_id: smiles}. IDs that fail or have no structure are omitted.
    Sleeps `sleep_between` seconds between requests to respect rate limits.
    """
    result: dict[str, str] = {}
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i : i + batch_size]
        ids_param = ";".join(batch)
        url = (
            f"{_CHEMBL_BASE}/molecule.json"
            f"?molecule_chembl_id__in={ids_param}"
            f"&only=molecule_chembl_id,molecule_structures"
            f"&limit={len(batch)}"
        )
        try:
            data = _get_json(url, timeout=timeout)
            for m in data.get("molecules", []):
                cid = m.get("molecule_chembl_id", "")
                structs = m.get("molecule_structures") or {}
                smi = structs.get("canonical_smiles", "")
                if cid and smi:
                    result[cid] = smi
        except Exception:
            pass
        if i + batch_size < len(chembl_ids):
            time.sleep(sleep_between)
    return result


def _parse_chembl_record(m: dict) -> ChEMBLCompound:
    structs = m.get("molecule_structures") or {}
    props   = m.get("molecule_properties") or {}
    return ChEMBLCompound(
        chembl_id      = m.get("molecule_chembl_id", ""),
        smiles         = structs.get("canonical_smiles", ""),
        name           = m.get("pref_name", "") or "",
        mw             = float(props.get("full_mwt") or 0),
        alogp          = float(props.get("alogp") or 0),
        hba            = int(props.get("hba") or 0),
        hbd            = int(props.get("hbd") or 0),
        psa            = float(props.get("psa") or 0),
        ro5_violations = int(props.get("num_ro5_violations") or 0),
    )


# ---------------------------------------------------------------------------
# ZINC
# ---------------------------------------------------------------------------

_ZINC_BASE = "https://zinc.docking.org"


@dataclass
class ZINCCompound:
    zinc_id: str
    smiles: str
    mw: float = 0.0
    logp: float = 0.0


def zinc_by_id(zinc_id: str, timeout: int = 15) -> ZINCCompound | None:
    """
    Fetch a single compound from ZINC20 by ID, e.g. 'ZINC000001234567'.
    Falls back to short form 'ZINC1234567' — the API accepts both.
    """
    url = f"{_ZINC_BASE}/substances/{zinc_id}.json"
    try:
        data = _get_json(url, timeout=timeout)
    except urllib.error.HTTPError:
        return None
    return _parse_zinc_record(data)


def zinc_smiles(
    zinc_ids: list[str],
    timeout: int = 15,
    sleep_between: float = 0.2,
) -> dict[str, str]:
    """Return {zinc_id: smiles} for a list of ZINC IDs."""
    result: dict[str, str] = {}
    for zid in zinc_ids:
        compound = zinc_by_id(zid, timeout=timeout)
        if compound and compound.smiles:
            result[zid] = compound.smiles
        time.sleep(sleep_between)
    return result


def zinc_subsets() -> list[str]:
    """List available ZINC20 subset names (tranches)."""
    url = f"{_ZINC_BASE}/tranches/catalogs.json"
    try:
        data = _get_json(url)
        return [t["name"] for t in data]
    except Exception:
        # Return well-known subset names as fallback
        return [
            "Drug-Like", "Lead-Like", "Fragment-Like",
            "Lugs", "Goldilocks", "Big-n-Greasy",
        ]


def zinc_download_tranche(
    tranche: str,
    max_mols: int = 1000,
    timeout: int = 60,
) -> list[str]:
    """
    Download SMILES from a named ZINC20 tranche.

    tranche: e.g. 'Drug-Like', 'Lead-Like', 'Fragment-Like'
    Returns up to `max_mols` SMILES strings.

    Note: ZINC tranche URLs are versioned — this targets ZINC20 (2021+).
    """
    # ZINC20 SMILES-only endpoint for a given catalog/tranche
    encoded = urllib.parse.quote(tranche)
    url = f"{_ZINC_BASE}/tranches/smiles.txt?tranche_name={encoded}&count={max_mols}"
    try:
        text = _get_text(url, timeout=timeout)
        smiles = [line.split()[0] for line in text.strip().splitlines() if line.strip()]
        return smiles[:max_mols]
    except Exception:
        return []


def zinc_random_sample(n: int = 100, mw_range: tuple[float, float] = (200, 500)) -> list[str]:
    """
    Return up to `n` drug-like SMILES from ZINC via the REST slice API.

    mw_range: (min_mw, max_mw) molecular weight filter.
    """
    lo, hi = mw_range
    url = (
        f"{_ZINC_BASE}/substances.json"
        f"?mwt_range={lo:.0f}-{hi:.0f}"
        f"&count={n}"
    )
    try:
        data = _get_json(url)
        substances = data if isinstance(data, list) else data.get("substances", [])
        return [s["smiles"] for s in substances if s.get("smiles")]
    except Exception:
        return []


def _parse_zinc_record(d: dict) -> ZINCCompound:
    return ZINCCompound(
        zinc_id = d.get("zinc_id", ""),
        smiles  = d.get("smiles", ""),
        mw      = float(d.get("mwt") or 0),
        logp    = float(d.get("logp") or 0),
    )


# ---------------------------------------------------------------------------
# TDC (Therapeutics Data Commons)
# ---------------------------------------------------------------------------

def tdc_dataset(
    name: str,
    split_method: str = "scaffold",
) -> "dict[str, pd.DataFrame]":
    """
    Load a TDC benchmark dataset and return train/valid/test splits.

    ``name`` can be any TDC ADMET single-prediction or DTI multi-prediction
    dataset, e.g.:
    - ADMET: ``"BBB_Martini"``, ``"hERG"``, ``"AMES"``, ``"Caco2_Wang"``
    - DTI:   ``"BindingDB_Kd"``, ``"BindingDB_IC50"``, ``"Davis"``, ``"KIBA"``

    Returns a dict with keys ``"train"``, ``"valid"``, ``"test"``.
    Each value is a pandas DataFrame. ADMET frames have columns
    ``["Drug", "Y"]``; DTI frames have ``["Drug", "Target", "Y"]``.

    Requires ``pip install molcore[bio]`` (PyTDC).
    """
    try:
        import tdc  # noqa: F401
    except ImportError:
        raise ImportError("PyTDC is required: pip install molcore[bio]")

    # Try ADMET single-pred first, fall back to DTI
    try:
        from tdc.single_pred import ADMET as TDC_ADMET
        data = TDC_ADMET(name=name)
    except Exception:
        try:
            from tdc.multi_pred import DTI as TDC_DTI
            data = TDC_DTI(name=name)
        except Exception as exc:
            raise ValueError(
                f"Could not load TDC dataset {name!r}. "
                f"Check the name against https://tdcommons.ai/. Original error: {exc}"
            )

    return data.get_split(method=split_method)


# ---------------------------------------------------------------------------
# BindingDB (via TDC)
# ---------------------------------------------------------------------------

@dataclass
class BindingRecord:
    smiles: str
    target_sequence: str
    affinity: float          # raw value in original units (nM for Kd/IC50/Ki)
    affinity_type: str       # "Kd", "IC50", "Ki", "EC50"
    target_name: str = ""


def bindingdb_search(
    affinity: str = "Kd",
    target: str | None = None,
    max_records: int = 5_000,
    split: str = "train",
) -> list[BindingRecord]:
    """
    Fetch drug-target binding records from BindingDB via TDC.

    Parameters
    ----------
    affinity : str
        One of ``"Kd"``, ``"IC50"``, ``"Ki"``, ``"EC50"``.
    target : str or None
        Optional substring filter on the target name/sequence column.
        Pass a UniProt ID fragment or protein name fragment to narrow results.
    max_records : int
        Maximum number of records to return.
    split : str
        TDC split to use: ``"train"``, ``"valid"``, or ``"test"``.

    Returns a list of :class:`BindingRecord` objects.

    Requires ``pip install molcore[bio]`` (PyTDC).
    """
    splits = tdc_dataset(f"BindingDB_{affinity}", split_method="random")
    df = splits[split]

    if target is not None:
        mask = (
            df["Target"].str.contains(target, case=False, na=False)
            if "Target_ID" not in df.columns
            else df.get("Target_ID", df["Target"]).str.contains(target, case=False, na=False)
        )
        df = df[mask]

    df = df.head(max_records)

    records = []
    for _, row in df.iterrows():
        records.append(BindingRecord(
            smiles          = str(row.get("Drug", "")),
            target_sequence = str(row.get("Target", "")),
            affinity        = float(row.get("Y", float("nan"))),
            affinity_type   = affinity,
            target_name     = str(row.get("Target_ID", "")),
        ))
    return records
