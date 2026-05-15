"""
molcore.admet — ADMET profiling for drug discovery.

Two layers:
  1. Rule-based filters (no extra deps) — Lipinski Ro5, Veber, Egan, PAINS, Brenk.
  2. ML predictors backed by TDC benchmark datasets (requires `pip install molcore[bio]`).

Quick start (rules only)::

    from molcore.admet import admet_screen
    df = admet_screen(["CC(=O)Oc1ccccc1C(=O)O", "CCO"])

ML prediction (requires PyTDC + scikit-learn)::

    from molcore.admet import ADMETPredictor
    pred = ADMETPredictor.from_tdc("BBB_Martini")   # downloads data, trains RF
    probs = pred.predict(smiles_list)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Supported TDC endpoints and whether they are classification tasks
# ---------------------------------------------------------------------------

TDC_CLASSIFICATION_ENDPOINTS = {
    "BBB_Martini",       # blood-brain barrier penetration
    "hERG",              # cardiac toxicity (ion channel)
    "AMES",              # Ames mutagenicity
    "CYP2C19_Veith",     # CYP2C19 inhibition
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "Bioavailability_Ma",
    "HIA_Hou",           # human intestinal absorption
    "Pgp_Broccatelli",   # P-glycoprotein inhibition
    "ClinTox",           # clinical toxicity
}

TDC_REGRESSION_ENDPOINTS = {
    "Caco2_Wang",        # Caco-2 membrane permeability
    "Solubility_AqSolDB",
    "HydrationFreeEnergy_FreeSolv",
    "Lipophilicity_AstraZeneca",
    "PPBR_AZ",           # plasma protein binding rate
    "VDss_Lombardo",     # volume of distribution
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
}


# ---------------------------------------------------------------------------
# Rule-based filters
# ---------------------------------------------------------------------------

def _rdmol(smiles: str):
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)


def _props(rdmol) -> dict:
    """Compute key physicochemical properties from an RDKit Mol."""
    from rdkit.Chem import Descriptors, rdMolDescriptors
    return {
        "mw":        Descriptors.MolWt(rdmol),
        "logp":      Descriptors.MolLogP(rdmol),
        "hbd":       rdMolDescriptors.CalcNumHBD(rdmol),
        "hba":       rdMolDescriptors.CalcNumHBA(rdmol),
        "tpsa":      Descriptors.TPSA(rdmol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(rdmol),
        "rings":     rdMolDescriptors.CalcNumRings(rdmol),
        "heavy_atoms": rdmol.GetNumHeavyAtoms(),
    }


def _pains_alerts(rdmol) -> list[str]:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    matches = catalog.GetMatches(rdmol)
    return [m.GetDescription() for m in matches]


def _brenk_alerts(rdmol) -> list[str]:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    matches = catalog.GetMatches(rdmol)
    return [m.GetDescription() for m in matches]


@dataclass
class ADMETProfile:
    """Per-molecule ADMET rule-based profile."""
    smiles: str
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    rot_bonds: int = 0
    rings: int = 0
    heavy_atoms: int = 0
    # rule outcomes
    lipinski_pass: bool = False     # Ro5: MW≤500, logP≤5, HBD≤5, HBA≤10
    veber_pass: bool = False        # RotBonds≤10, TPSA≤140
    egan_pass: bool = False         # logP≤5.88, TPSA≤131.6
    druglike: bool = False          # Lipinski + Veber + no PAINS
    pains_alerts: list = field(default_factory=list)
    brenk_alerts: list = field(default_factory=list)
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "smiles":        self.smiles,
            "mw":            self.mw,
            "logp":          self.logp,
            "hbd":           self.hbd,
            "hba":           self.hba,
            "tpsa":          self.tpsa,
            "rot_bonds":     self.rot_bonds,
            "rings":         self.rings,
            "heavy_atoms":   self.heavy_atoms,
            "lipinski_pass": self.lipinski_pass,
            "veber_pass":    self.veber_pass,
            "egan_pass":     self.egan_pass,
            "druglike":      self.druglike,
            "n_pains":       len(self.pains_alerts),
            "n_brenk":       len(self.brenk_alerts),
            "parse_error":   self.parse_error,
        }


def _profile_one(smiles: str) -> ADMETProfile:
    rdmol = _rdmol(smiles)
    if rdmol is None:
        return ADMETProfile(smiles=smiles, parse_error=True)

    p = _props(rdmol)
    lipinski = (
        p["mw"] <= 500
        and p["logp"] <= 5
        and p["hbd"] <= 5
        and p["hba"] <= 10
    )
    veber = p["rot_bonds"] <= 10 and p["tpsa"] <= 140
    egan  = p["logp"] <= 5.88 and p["tpsa"] <= 131.6
    pains = _pains_alerts(rdmol)
    brenk = _brenk_alerts(rdmol)

    return ADMETProfile(
        smiles       = smiles,
        mw           = round(p["mw"], 2),
        logp         = round(p["logp"], 3),
        hbd          = p["hbd"],
        hba          = p["hba"],
        tpsa         = round(p["tpsa"], 2),
        rot_bonds    = p["rot_bonds"],
        rings        = p["rings"],
        heavy_atoms  = p["heavy_atoms"],
        lipinski_pass = lipinski,
        veber_pass   = veber,
        egan_pass    = egan,
        druglike     = lipinski and veber and len(pains) == 0,
        pains_alerts = pains,
        brenk_alerts = brenk,
    )


def admet_screen(smiles: list[str]) -> "list[ADMETProfile]":
    """
    Rule-based ADMET screen for a list of SMILES.

    Returns one :class:`ADMETProfile` per molecule. No extra dependencies
    beyond RDKit (already required by molcore).

    Use :func:`admet_screen_df` for a pandas DataFrame view.

    Example::

        profiles = admet_screen(["CC(=O)Oc1ccccc1C(=O)O", "CCO"])
        for p in profiles:
            print(p.smiles, p.lipinski_pass, p.druglike)
    """
    return [_profile_one(smi) for smi in smiles]


def admet_screen_df(smiles: list[str]) -> "pd.DataFrame":
    """
    Rule-based ADMET screen. Returns a pandas DataFrame (one row per molecule).

    Columns: smiles, mw, logp, hbd, hba, tpsa, rot_bonds, rings, heavy_atoms,
    lipinski_pass, veber_pass, egan_pass, druglike, n_pains, n_brenk, parse_error.
    """
    import pandas as _pd
    profiles = admet_screen(smiles)
    return _pd.DataFrame([p.to_dict() for p in profiles])


# ---------------------------------------------------------------------------
# ML-based ADMET predictor (requires pip install molcore[bio])
# ---------------------------------------------------------------------------

class ADMETPredictor:
    """
    Random-forest ADMET endpoint predictor trained on TDC benchmark data.

    Classification endpoints return probabilities (float in [0, 1]).
    Regression endpoints return raw predicted values.

    Requires ``pip install molcore[bio]`` (PyTDC + scikit-learn).

    Usage::

        pred = ADMETPredictor.from_tdc("BBB_Martini")
        probs = pred.predict(["CC(=O)Oc1ccccc1C(=O)O", "CCO"])
        # array([0.82, 0.24])
    """

    def __init__(
        self,
        endpoint: str,
        model,                   # sklearn estimator
        is_classification: bool,
        nbits: int = 2048,
        radius: int = 2,
    ):
        self.endpoint = endpoint
        self._model = model
        self.is_classification = is_classification
        self._nbits = nbits
        self._radius = radius

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tdc(
        cls,
        endpoint: str,
        n_estimators: int = 300,
        nbits: int = 2048,
        radius: int = 2,
        split_method: str = "scaffold",
        random_state: int = 42,
    ) -> "ADMETPredictor":
        """
        Download a TDC ADMET dataset, train a random forest, return a ready
        predictor.

        ``endpoint`` must be one of the TDC ADMET benchmark names, e.g.
        ``"BBB_Martini"``, ``"hERG"``, ``"AMES"``, ``"Caco2_Wang"``.

        The full list of supported names is in :data:`TDC_CLASSIFICATION_ENDPOINTS`
        and :data:`TDC_REGRESSION_ENDPOINTS`.
        """
        try:
            from tdc.single_pred import ADMET as TDC_ADMET
        except ImportError:
            raise ImportError(
                "PyTDC is required: pip install molcore[bio]"
            )
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required: pip install molcore[bio]"
            )

        data  = TDC_ADMET(name=endpoint)
        split = data.get_split(method=split_method)

        train_df = split["train"]
        is_clf = endpoint in TDC_CLASSIFICATION_ENDPOINTS

        train_fps = _morgan_matrix(train_df["Drug"].tolist(), nbits=nbits, radius=radius)
        train_y   = train_df["Y"].to_numpy()

        if is_clf:
            model = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, n_jobs=-1, random_state=random_state
            )

        model.fit(train_fps, train_y)
        return cls(endpoint, model, is_clf, nbits=nbits, radius=radius)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, smiles: list[str]) -> np.ndarray:
        """
        Predict the ADMET endpoint for each SMILES.

        Classification: returns positive-class probability (float in [0,1]).
        Regression: returns raw predicted value.

        Invalid SMILES get ``np.nan``.
        """
        fps = _morgan_matrix(smiles, nbits=self._nbits, radius=self._radius)
        valid_mask = ~np.isnan(fps).any(axis=1)

        result = np.full(len(smiles), np.nan)

        if valid_mask.any():
            valid_fps = fps[valid_mask]
            if self.is_classification:
                preds = self._model.predict_proba(valid_fps)[:, 1]
            else:
                preds = self._model.predict(valid_fps)
            result[valid_mask] = preds

        return result

    def predict_with_threshold(
        self,
        smiles: list[str],
        threshold: float = 0.5,
    ) -> list[bool | None]:
        """
        For classification endpoints: return True/False/None (None for invalid SMILES).
        """
        if not self.is_classification:
            raise ValueError("predict_with_threshold is only for classification endpoints")
        probs = self.predict(smiles)
        return [None if np.isnan(p) else bool(p >= threshold) for p in probs]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "str | pathlib.Path") -> None:
        """Serialize to a pickle file (sklearn convention)."""
        import pickle
        import pathlib
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: "str | pathlib.Path") -> "ADMETPredictor":
        """Load a previously saved ADMETPredictor."""
        import pickle
        import pathlib
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ADMETPredictor, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        kind = "classifier" if self.is_classification else "regressor"
        return f"ADMETPredictor(endpoint={self.endpoint!r}, type={kind})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _morgan_matrix(
    smiles: list[str],
    nbits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """Compute Morgan fingerprints as uint8 array (N, nbits). NaN row on failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.full(nbits, np.nan))
        else:
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            arr = np.zeros(nbits, dtype=np.uint8)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            rows.append(arr.astype(np.float32))

    return np.vstack(rows)
