import argparse
import json
import os
import resource
import typing as t

import torch
from transformers import AutoModel, AutoTokenizer


def get_peak_rss() -> int:
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def get_gpu_usage() -> list[dict[str, t.Any]]:
    usage: list[dict[str, t.Any]] = []

    if torch.cuda.is_available():
        # for each GPU
        for i in range(torch.cuda.device_count()):
            dev = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            (free, total) = mem

            usage.append(
                {
                    "device_index": i,
                    "device_name": dev.name,
                    "total_memory": total,
                    "free_memory": free,
                }
            )

    return usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile model files")
    parser.add_argument("--model", help="Path to HF model directory", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    args = parser.parse_args()

    path: str = os.path.abspath(args.model)
    inputs: t.Any | None = None

    errors: dict[str, str | None] = {
        "tokenizer": None,
        "model": None,
    }

    ram: dict[str, int] = {"start": get_peak_rss()}
    gpu: dict[str, list[dict[str, t.Any]]] = {"start": get_gpu_usage()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(
        json.dumps(
            {
                "ram": ram,
                "gpu": gpu,
                "errors": errors,
            }
        )
    )
