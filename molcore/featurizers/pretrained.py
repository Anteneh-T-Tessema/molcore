"""
Pretrained model wrappers: ChemBERTa, ChemGPT, Graphormer.
Requires: pip install molfeat
"""
import torch


def chemberta(smiles: list[str], model: str = "DeepChem/ChemBERTa-77M-MLM") -> torch.Tensor:
    from molfeat.trans.pretrained import PretrainedHFTranslator
    t = PretrainedHFTranslator(kind=model)
    return torch.tensor(t(smiles), dtype=torch.float32)


def graphormer(smiles: list[str]) -> torch.Tensor:
    from molfeat.trans.pretrained.graphormer import GraphormerTranslator
    t = GraphormerTranslator()
    return torch.tensor(t(smiles), dtype=torch.float32)
