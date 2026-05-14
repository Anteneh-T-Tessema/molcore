"""
Pretrained model wrappers: ChemBERTa, MolT5, Graphormer.

ChemBERTa / Graphormer: pip install molfeat
MolT5               : pip install transformers sentencepiece
"""
from __future__ import annotations
import torch


def chemberta(smiles: list[str], model: str = "DeepChem/ChemBERTa-77M-MLM") -> torch.Tensor:
    from molfeat.trans.pretrained import PretrainedHFTranslator
    t = PretrainedHFTranslator(kind=model)
    return torch.tensor(t(smiles), dtype=torch.float32)


def graphormer(smiles: list[str]) -> torch.Tensor:
    from molfeat.trans.pretrained.graphormer import GraphormerTranslator
    t = GraphormerTranslator()
    return torch.tensor(t(smiles), dtype=torch.float32)


def molt5_encode(
    smiles: list[str],
    model: str = "laituan245/molt5-small",
    max_length: int = 512,
    device: str | None = None,
) -> torch.Tensor:
    """
    Encode SMILES strings with MolT5 (encoder only).

    Returns mean-pooled last-hidden-state: shape (N, hidden_size).
    hidden_size = 512 for molt5-small, 768 for molt5-base, 1024 for molt5-large.

    Requires: pip install transformers sentencepiece
    """
    from transformers import T5Tokenizer, T5EncoderModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(model, model_max_length=max_length)
    encoder   = T5EncoderModel.from_pretrained(model).to(device).eval()

    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        hidden = encoder(**inputs).last_hidden_state  # (N, L, H)

    # Mean-pool over sequence length, ignoring padding
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (N, L, 1)
    embeddings = (hidden * mask).sum(1) / mask.sum(1)      # (N, H)
    return embeddings.cpu()


def molt5_translate(
    smiles: list[str],
    task: str = "SMILES2IUPAC",
    model: str = "laituan245/molt5-small-smiles2iupac",
    max_new_tokens: int = 128,
    device: str | None = None,
) -> list[str]:
    """
    Run a MolT5 seq2seq translation task.

    task="SMILES2IUPAC" : SMILES → IUPAC name (default model: molt5-small-smiles2iupac)
    task="IUPAC2SMILES" : IUPAC  → SMILES       (use molt5-small-iupac2smiles)

    Returns a list of decoded strings, one per input.
    Requires: pip install transformers sentencepiece
    """
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(model)
    model_obj  = T5ForConditionalGeneration.from_pretrained(model).to(device).eval()

    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model_obj.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
