#!/usr/bin/env python
"""Export a USER-bge-m3 query encoder to ONNX with fused transformer kernels.

The exported graph returns the L2-normalized CLS vector, exactly what
SentenceTransformer(USER-bge-m3).encode(..., normalize_embeddings=True) returns
(CLS pooling + Normalize). ONNX Runtime's transformer optimizer then fuses
attention, bias+GELU and skip+LayerNorm; weights stay fp32, so the output
matches PyTorch to ~1e-6 (checked in notebook 28). No quantization is applied:
the recall of the evaluated system must not change.

`--model` accepts a Hub id or a local directory, so the merged LoRA v2 model
(Kaggle stage 10A output) exports with the same command. Output (~1.4 GB) goes
to `artifacts/onnx/` and is not tracked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepvk/USER-bge-m3"
MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"


class NormalizedCLS(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(hidden, dim=-1)


def export(model_name: str, revision: str | None, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).eval()
    example = tokenizer(["пример запроса"], return_tensors="pt")
    raw = output / "encoder.onnx"
    torch.onnx.export(
        NormalizedCLS(model), (example["input_ids"], example["attention_mask"]), str(raw),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False, external_data=True,
    )
    config = model.config
    optimized = optimize_model(str(raw), model_type="bert", num_heads=config.num_attention_heads,
                               hidden_size=config.hidden_size)
    print(optimized.get_fused_operator_statistics())
    target = output / "encoder_opt.onnx"
    optimized.save_model_to_file(str(target), use_external_data_format=True)
    tokenizer.save_pretrained(output)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/onnx/user_bge_m3")
    args = parser.parse_args()
    print("saved", export(args.model, args.revision or None, args.output))


if __name__ == "__main__":
    main()
