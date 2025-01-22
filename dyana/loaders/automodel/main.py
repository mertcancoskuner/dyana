import argparse
import json
import os
import typing as t

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model files")
    parser.add_argument("--model", help="Path to HF model directory", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    parser.add_argument("--low-memory", action="store_true", help="Use low memory mode")
    args = parser.parse_args()

    path: str = os.path.abspath(args.model)
    inputs: t.Any | None = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler: Profiler = Profiler(gpu=True)

    if args.low_memory:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    try:
        if args.low_memory:
            # initialize tokenizer structure with empty weights allocated on
            # a meta torch device
            with init_empty_weights(include_buffers=True):
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                profiler.track_memory("after_tokenizer_initialized")
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            profiler.track_memory("after_tokenizer_loaded")

            inputs = tokenizer(args.input, return_tensors="pt").to(device)
            profiler.track_memory("after_tokenization")

    except Exception as e:
        profiler.track_error("tokenizer", str(e))

    try:
        if args.low_memory:
            # initialize model structure with empty weights allocated on
            # a meta torch device
            with init_empty_weights(include_buffers=True):
                model = AutoModel.from_config(config, trust_remote_code=True)
                profiler.track_memory("after_model_initialized")
        else:
            if inputs is None:
                raise ValueError("tokenization failed")

            # load model weights and perform inference
            model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
            profiler.track_memory("after_model_loaded")

            # no need to compute gradients
            with torch.no_grad():
                outputs = model(**inputs)
                profiler.track_memory("after_model_inference")

    except Exception as e:
        profiler.track_error("model", str(e))

    print(json.dumps(profiler.as_dict()))
