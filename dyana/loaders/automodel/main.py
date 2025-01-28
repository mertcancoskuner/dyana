import argparse
import logging
import os
import typing as t

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    logging.disable(logging.ERROR)

    parser = argparse.ArgumentParser(description="Profile model files")
    parser.add_argument("--model", help="Path to HF model directory", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    parser.add_argument("--low-memory", action="store_true", help="Use low memory mode")
    args = parser.parse_args()

    path: str = os.path.abspath(args.model)
    inputs: t.Any | None = None
    profiler: Profiler = Profiler(gpu=True)
    has_tokenizer: bool = os.path.exists(os.path.join(path, "tokenizer.json"))

    if args.low_memory:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    if has_tokenizer:
        try:
            if args.low_memory:
                # initialize tokenizer structure with empty weights allocated on
                # a meta torch device
                with init_empty_weights(include_buffers=True):
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    profiler.on_stage("after_tokenizer_initialized")
            else:
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                profiler.on_stage("after_tokenizer_loaded")

                inputs = tokenizer(args.input, return_tensors="pt")
                profiler.on_stage("after_tokenization")

        except Exception as e:
            profiler.track_error("tokenizer", str(e))

    try:
        if args.low_memory:
            # initialize model structure with empty weights allocated on
            # a meta torch device
            with init_empty_weights(include_buffers=True):
                model = AutoModel.from_config(config, trust_remote_code=True)
                profiler.on_stage("after_model_initialized")
        else:
            # load model weights and perform inference
            model = AutoModel.from_pretrained(path, trust_remote_code=True, device_map="auto")
            profiler.on_stage("after_model_loaded")

            if has_tokenizer:
                if inputs is None:
                    raise ValueError("tokenization failed")

                # no need to compute gradients
                with torch.no_grad():
                    outputs = model(**inputs)
                    profiler.on_stage("after_model_inference")
            else:
                profiler.track_warning("model", "tokenizer not found, inference skipped")

    except Exception as e:
        profiler.track_error("model", str(e))
