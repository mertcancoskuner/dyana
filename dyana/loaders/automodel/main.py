import argparse
import json
import os
import typing as t

import torch
from transformers import AutoModel, AutoTokenizer

from dyana import get_current_imports, get_gpu_usage, get_peak_rss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model files")
    parser.add_argument("--model", help="Path to HF model directory", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    args = parser.parse_args()

    path: str = os.path.abspath(args.model)
    inputs: t.Any | None = None
    errors: dict[str, str] = {}
    ram: dict[str, int] = {"start": get_peak_rss()}
    gpu: dict[str, list[dict[str, t.Any]]] = {"start": get_gpu_usage()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imports_at_start = get_current_imports()

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        ram["after_tokenizer_loaded"] = get_peak_rss()
        gpu["after_tokenizer_loaded"] = get_gpu_usage()
        inputs = tokenizer(args.input, return_tensors="pt").to(device)
        ram["after_tokenization"] = get_peak_rss()
        gpu["after_tokenization"] = get_gpu_usage()
    except Exception as e:
        errors["tokenizer"] = str(e)

    try:
        if inputs is None:
            raise ValueError("tokenization failed")

        model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
        ram["after_model_loaded"] = get_peak_rss()
        gpu["after_model_loaded"] = get_gpu_usage()

        # no need to compute gradients
        with torch.no_grad():
            outputs = model(**inputs)
            ram["after_model_inference"] = get_peak_rss()
            gpu["after_model_inference"] = get_gpu_usage()

    except Exception as e:
        errors["model"] = str(e)

    imports_at_end = get_current_imports()
    imported = {k: imports_at_end[k] for k in imports_at_end if k not in imports_at_start}

    print(
        json.dumps(
            {
                "ram": ram,
                "gpu": gpu,
                "errors": errors,
                "extra": {
                    "imports": imported,
                },
            }
        )
    )
