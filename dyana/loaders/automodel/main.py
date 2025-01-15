import argparse
import json
import os
import typing as t

import torch
from transformers import AutoModel, AutoTokenizer

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model files")
    parser.add_argument("--model", help="Path to HF model directory", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    args = parser.parse_args()

    path: str = os.path.abspath(args.model)
    inputs: t.Any | None = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler: Profiler = Profiler(gpu=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        profiler.track_memory("after_tokenizer_loaded")

        inputs = tokenizer(args.input, return_tensors="pt").to(device)
        profiler.track_memory("after_tokenization")

    except Exception as e:
        profiler.track_error("tokenizer", str(e))

    try:
        if inputs is None:
            raise ValueError("tokenization failed")

        model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
        profiler.track_memory("after_model_loaded")

        # no need to compute gradients
        with torch.no_grad():
            outputs = model(**inputs)
            profiler.track_memory("after_model_inference")

    except Exception as e:
        profiler.track_error("model", str(e))

    print(json.dumps(profiler.as_dict()))
