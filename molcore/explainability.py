"""
Atom/bond attribution — maps tensor gradients back to graph nodes.
Works with any differentiable model that takes PyG Data as input.
"""
import torch
from torch_geometric.data import Data


def atom_attribution(model, data: Data, target_class: int = 0) -> torch.Tensor:
    """
    Gradient-based atom importance scores.
    Returns (N,) float tensor — one score per atom.
    """
    data.x.requires_grad_(True)
    out = model(data)
    score = out[0, target_class] if out.dim() > 1 else out[0]
    score.backward()
    return data.x.grad.abs().sum(dim=-1)  # (N,) — sum over feature dim


def integrated_gradients(
    model,
    data: Data,
    steps: int = 50,
    target_class: int = 0,
) -> torch.Tensor:
    """Integrated gradients attribution — (N,) float tensor."""
    baseline = torch.zeros_like(data.x)
    attributions = torch.zeros(data.x.shape[0])

    for alpha in torch.linspace(0, 1, steps):
        interp = Data(
            x          = baseline + alpha * (data.x - baseline),
            edge_index = data.edge_index,
            edge_attr  = data.edge_attr,
        )
        interp.x.requires_grad_(True)
        out = model(interp)
        score = out[0, target_class] if out.dim() > 1 else out[0]
        score.backward()
        attributions += interp.x.grad.abs().sum(dim=-1).detach()

    return attributions / steps
