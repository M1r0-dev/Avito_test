"""ONNX Runtime query encoder, a drop-in for SentenceTransformer.encode on one text.

Built by `scripts/export_onnx_encoder.py`: same weights (fp32), CLS pooling and
L2 normalization as USER-bge-m3; transformer kernels fused by ONNX Runtime.
Notebook 28 checks that its vectors give the same rankings as PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MAX_LENGTH = 256  # queries are short (p95 26 tokens); the Kaggle kernels encoded with 256


class OnnxQueryEncoder:
    def __init__(self, directory: Path, threads: int, allow_spinning: bool = True) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # Spinning trades idle CPU for latency; with several concurrent branches
        # it steals cores from the other branch, so the online path disables it.
        options.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(directory / "encoder_opt.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory)

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text], return_tensors="np", truncation=True, max_length=MAX_LENGTH)
        return self.session.run(None, {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        })[0][0]
