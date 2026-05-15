"""
Tests for bioinformatics features:
  - admet_screen / ADMETProfile (rule-based, no extra deps)
  - ProteinSeq construction and FASTA parsing
  - ProteinSeq.to_pyg()
  - ESM-2 embedding (skipped if transformers not installed)
  - ADMETPredictor.from_tdc (skipped if PyTDC not installed)
  - MolDataset.from_tdc (skipped if PyTDC not installed)
"""
import math
import textwrap

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

ASPIRIN  = "CC(=O)Oc1ccccc1C(=O)O"   # MW=180, logP=1.19 — passes Lipinski
ETHANOL  = "CCO"                       # tiny, passes everything
CYCLOSPORINE = "CCC1NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(C)NC(=O)C(NC(=O)C(CC(C)C)N(C)C1=O)C(C)CC"
# cyclosporin A MW≈1202 — violates Lipinski

PAINS_EXAMPLE = "O=C1c2ccccc2C(=O)c2ccccc21"  # anthraquinone — hits PAINS
INVALID_SMILES = "not_a_smiles"

# ---------------------------------------------------------------------------
# ADMET rule-based
# ---------------------------------------------------------------------------

def test_admet_screen_aspirin():
    from molcore.admet import admet_screen
    profiles = admet_screen([ASPIRIN])
    assert len(profiles) == 1
    p = profiles[0]
    assert p.lipinski_pass
    assert p.veber_pass
    assert not p.parse_error
    assert p.mw == pytest.approx(180.16, abs=0.5)


def test_admet_screen_cyclosporine_fails_lipinski():
    from molcore.admet import admet_screen
    profiles = admet_screen([CYCLOSPORINE])
    p = profiles[0]
    assert not p.lipinski_pass
    assert p.mw > 500


def test_admet_screen_pains_detected():
    from molcore.admet import admet_screen
    profiles = admet_screen([PAINS_EXAMPLE])
    p = profiles[0]
    assert len(p.pains_alerts) > 0
    assert not p.druglike


def test_admet_screen_invalid_smiles():
    from molcore.admet import admet_screen
    profiles = admet_screen([INVALID_SMILES])
    assert profiles[0].parse_error


def test_admet_screen_batch():
    from molcore.admet import admet_screen
    smiles = [ASPIRIN, ETHANOL, CYCLOSPORINE, INVALID_SMILES]
    profiles = admet_screen(smiles)
    assert len(profiles) == 4
    assert not profiles[3].lipinski_pass  # invalid → parse_error


def test_admet_screen_df_columns():
    pytest.importorskip("pandas")
    from molcore.admet import admet_screen_df
    df = admet_screen_df([ASPIRIN, ETHANOL])
    expected = {
        "smiles", "mw", "logp", "hbd", "hba", "tpsa",
        "rot_bonds", "lipinski_pass", "veber_pass", "egan_pass",
        "druglike", "n_pains", "n_brenk", "parse_error",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == 2


def test_admet_druglike_flags():
    from molcore.admet import admet_screen
    p = admet_screen([ASPIRIN])[0]
    assert p.druglike


def test_admet_profile_to_dict():
    from molcore.admet import admet_screen
    p = admet_screen([ASPIRIN])[0]
    d = p.to_dict()
    assert isinstance(d, dict)
    assert "smiles" in d
    assert "mw" in d
    assert "lipinski_pass" in d
    assert "druglike" in d


def test_admet_profile_to_dict_invalid():
    from molcore.admet import admet_screen
    p = admet_screen([INVALID_SMILES])[0]
    d = p.to_dict()
    assert d["parse_error"] is True


# ---------------------------------------------------------------------------
# ProteinSeq — construction
# ---------------------------------------------------------------------------

SEQ_SHORT  = "MKTLLILAVLCLGFAQAS"
SEQ_LONGER = "ACDEFGHIKLMNPQRSTVWY" * 5   # 100 residues

def test_protein_from_sequence_basic():
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence(SEQ_SHORT, name="signal")
    assert len(p) == len(SEQ_SHORT)
    assert p.name == "signal"
    assert p.sequence == SEQ_SHORT.upper()


def test_protein_from_sequence_normalises_case():
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence("mktlli")
    assert p.sequence == "MKTLLI"


def test_protein_from_sequence_rejects_empty():
    from molcore.protein import ProteinSeq
    with pytest.raises(ValueError, match="empty"):
        ProteinSeq.from_sequence("")


def test_protein_from_sequence_rejects_nonstandard():
    from molcore.protein import ProteinSeq
    with pytest.raises(ValueError, match="Non-standard"):
        ProteinSeq.from_sequence("MKTX123")   # digits + X not in standard 20


def test_protein_from_sequence_rejects_null_byte_in_length():
    from molcore.protein import ProteinSeq
    # 100 001 AA — over the 100 000 limit
    with pytest.raises(ValueError, match="exceeds"):
        ProteinSeq.from_sequence("A" * 100_001)


# ---------------------------------------------------------------------------
# ProteinSeq — FASTA parsing
# ---------------------------------------------------------------------------

FASTA_TEXT = textwrap.dedent("""\
    >sp|P00533|EGFR_HUMAN Epidermal growth factor receptor
    MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA
    >sp|P42336|PK3CA_HUMAN Phosphatidylinositol 4,5-bisphosphate 3-kinase
    MPPRPSSGELWGIHLMPPRILVECLLPNGMIVTLECLREATLISNEEAKNIINEVHSSINDTMHIQQQFALQNAKMESVGTPMFNKTTCNISNTGSGQIIVHDSISCPDGQFLLENMLEQNQHISANISENSFEELHKSIIAQNEELARRSESEIRTEEYELLNDRLFAMQSEKPGDPESANRTSSREQKLISEEDL
""")

def test_fasta_parse_returns_list():
    from molcore.protein import ProteinSeq
    seqs = ProteinSeq.from_fasta_string(FASTA_TEXT)
    assert len(seqs) == 2


def test_fasta_parse_names():
    from molcore.protein import ProteinSeq
    seqs = ProteinSeq.from_fasta_string(FASTA_TEXT)
    assert "EGFR_HUMAN" in seqs[0].name or "P00533" in seqs[0].name


def test_fasta_parse_sequence_content():
    from molcore.protein import ProteinSeq
    seqs = ProteinSeq.from_fasta_string(FASTA_TEXT)
    assert seqs[0].sequence.startswith("MRPSG")


def test_fasta_parse_file(tmp_path):
    from molcore.protein import ProteinSeq
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(FASTA_TEXT)
    seqs = ProteinSeq.from_fasta(fasta_file)
    assert len(seqs) == 2


def test_fasta_file_not_found():
    from molcore.protein import ProteinSeq
    with pytest.raises(FileNotFoundError):
        ProteinSeq.from_fasta("/nonexistent/path/proteins.fasta")


# ---------------------------------------------------------------------------
# ProteinSeq — PyG graph
# ---------------------------------------------------------------------------

def test_protein_to_pyg_basic():
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence("ACDEF")
    data = p.to_pyg()
    assert data.x.shape == (5, 20)           # 5 residues, 20-dim one-hot
    assert data.edge_index.shape[0] == 2
    # bidirectional sequential edges: 2 × (L-1) = 2 × 4 = 8
    assert data.edge_index.shape[1] == 8


def test_protein_to_pyg_one_hot_correct():
    import torch
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence("A")
    data = p.to_pyg()
    # "A" is index 0 in ACDEFGHIKLMNPQRSTVWY
    assert data.x[0, 0].item() == 1.0
    assert data.x[0, 1:].sum().item() == 0.0


def test_protein_to_pyg_no_edges_single_residue():
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence("M")
    data = p.to_pyg()
    assert data.edge_index.shape[1] == 0


# ---------------------------------------------------------------------------
# ESM-2 embedding (skipped if transformers not installed)
# ---------------------------------------------------------------------------

def test_esm2_embed_returns_tensor():
    pytest.importorskip("transformers", reason="transformers not installed")
    import torch
    from molcore.protein import ProteinSeq
    p = ProteinSeq.from_sequence("MKTLLI")
    emb = p.embed(model="facebook/esm2_t6_8M_UR50D", pooling="mean", device="cpu")
    assert emb.ndim == 1
    assert emb.shape[0] == 320   # ESM-2 t6 hidden size


def test_esm2_embed_batch():
    pytest.importorskip("transformers", reason="transformers not installed")
    import torch
    from molcore.protein import ProteinSeq
    seqs = ["MKTLLI", "ACDEFG", "WYVNPR"]
    embs = ProteinSeq.embed_batch(
        seqs, model="facebook/esm2_t6_8M_UR50D", pooling="mean", device="cpu"
    )
    assert embs.shape == (3, 320)


def test_esm2_embed_per_residue():
    pytest.importorskip("transformers", reason="transformers not installed")
    import torch
    from molcore.protein import ProteinSeq
    seq = "MKTLLI"
    result = ProteinSeq.embed_batch(
        [seq], model="facebook/esm2_t6_8M_UR50D", pooling="none", device="cpu"
    )
    assert isinstance(result, list)
    # Length includes CLS + EOS tokens stripped by our mask, or full seq length
    assert result[0].shape[1] == 320


# ---------------------------------------------------------------------------
# ADMETPredictor + MolDataset.from_tdc (skipped if PyTDC not installed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bbb_predictor():
    pytest.importorskip("tdc", reason="PyTDC not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    from molcore.admet import ADMETPredictor
    return ADMETPredictor.from_tdc("BBB_Martini", n_estimators=50)


def test_admet_predictor_predict_shape(bbb_predictor):
    smiles = [ASPIRIN, ETHANOL, CYCLOSPORINE]
    probs = bbb_predictor.predict(smiles)
    assert probs.shape == (3,)
    assert all(0.0 <= p <= 1.0 or math.isnan(p) for p in probs)


def test_admet_predictor_invalid_smiles_gets_nan(bbb_predictor):
    probs = bbb_predictor.predict([INVALID_SMILES])
    assert math.isnan(probs[0])


def test_admet_predictor_threshold(bbb_predictor):
    results = bbb_predictor.predict_with_threshold([ASPIRIN, ETHANOL], threshold=0.5)
    assert all(r in (True, False, None) for r in results)


def test_admet_predictor_save_load(bbb_predictor, tmp_path):
    path = tmp_path / "bbb.pkl"
    bbb_predictor.save(path)
    from molcore.admet import ADMETPredictor
    loaded = ADMETPredictor.load(path)
    assert loaded.endpoint == bbb_predictor.endpoint
    orig  = bbb_predictor.predict([ASPIRIN])
    reloaded = loaded.predict([ASPIRIN])
    np.testing.assert_allclose(orig, reloaded, rtol=1e-5)


def test_moldataset_from_tdc_bbb():
    pytest.importorskip("tdc", reason="PyTDC not installed")
    from molcore.io import MolDataset
    ds = MolDataset.from_tdc("BBB_Martini", split="train", compute_fps=False, compute_desc=False)
    assert len(ds) > 0
    assert ds.labels is not None
    assert set(ds.labels).issubset({0.0, 1.0})   # binary classification


def test_moldataset_from_tdc_metadata():
    pytest.importorskip("tdc", reason="PyTDC not installed")
    from molcore.io import MolDataset
    ds = MolDataset.from_tdc("BBB_Martini", split="train", compute_fps=False, compute_desc=False)
    assert "tdc_dataset" in ds.metadata
    assert ds.metadata["tdc_dataset"][0] == "BBB_Martini"


def test_moldataset_from_bindingdb():
    pytest.importorskip("tdc", reason="PyTDC not installed")
    from molcore.io import MolDataset
    ds = MolDataset.from_bindingdb(
        affinity="Kd", max_records=100,
        compute_fps=False, compute_desc=False,
    )
    assert len(ds) > 0
    assert ds.labels is not None
    assert "protein_sequence" in ds.metadata
    # pIC50 values should be in a reasonable range
    valid_labels = ds.labels[~np.isnan(ds.labels)]
    assert valid_labels.mean() > 0
