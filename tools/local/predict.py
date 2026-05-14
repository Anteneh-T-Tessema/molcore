"""
tools/local/predict.py — GCN property prediction tool.

Modes:
  predict              : run inference on a list of SMILES
  predict_uncertainty  : MC Dropout mean + std
"""
from __future__ import annotations
import argparse
import json


def run(
    smiles: list[str],
    model_path: str,
    mode: str = "predict",
    n_samples: int = 30,
    device: str = "auto",
) -> dict:
    from molcore.predictor import PropertyPredictor

    pred = PropertyPredictor.load(model_path, device=device)

    if mode == "predict":
        preds = pred.predict(smiles)
        return {"predictions": preds.tolist()}

    if mode == "predict_uncertainty":
        mean, std = pred.predict_with_uncertainty(smiles, n_samples=n_samples)
        return {"mean": mean.tolist(), "std": std.tolist()}

    raise ValueError(f"Unknown mode: {mode!r}. Choose predict | predict_uncertainty")


def _cli() -> None:
    p = argparse.ArgumentParser(description="GCN property prediction")
    p.add_argument("--smiles", required=True, help="JSON list of SMILES strings")
    p.add_argument("--model", required=True, help="Path to .pt checkpoint")
    p.add_argument("--mode", default="predict",
                   choices=["predict", "predict_uncertainty"])
    p.add_argument("--n-samples", type=int, default=30,
                   help="MC Dropout samples (predict_uncertainty only)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    smiles = json.loads(args.smiles)
    result = run(
        smiles=smiles,
        model_path=args.model,
        mode=args.mode,
        n_samples=args.n_samples,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
