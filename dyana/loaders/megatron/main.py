import argparse
import logging
import sys
import warnings
import multiprocessing
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

multiprocessing.set_start_method("spawn", force=True)

import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import LlamaTokenizer

from dyana import Profiler


def find_tokenizer(model_path: Path) -> Path:
    """Find tokenizer file in model directory or alongside model file."""
    patterns = [
        # LLaMA specific patterns first
        "llama*tokenizer*.model",  # LLaMA specific naming
        "tokenizer.model",  # Standard LLaMA tokenizer
        # Generic patterns as fallback
        "*.model",  # sentencepiece models
        "tokenizer.*",  # huggingface style
        "*/tokenizer.*",  # nested folder
        "vocab.*",  # vocabulary files
        "merges.txt",  # BPE merges
    ]

    # Try both the model's directory and its parent directory
    search_dirs = [model_path.parent]
    if model_path.parent.parent.exists():
        search_dirs.append(model_path.parent.parent)

    for directory in search_dirs:
        all_files = list(directory.glob("*"))
        for f in sorted(all_files):
            print(f"  {f}", file=sys.stderr)
            # If it looks like a LLaMA tokenizer file, try it first
            if "tokenizer" in f.name.lower() and f.name.endswith(".model"):
                return f

        # If no obvious tokenizer found, try the patterns
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"No tokenizer found in {[str(d) for d in search_dirs]} after trying patterns: {patterns}\n"
        f"Available files in {model_path.parent}: {[f.name for f in model_path.parent.glob('*')]}"
    )


def load_tokenizer(args) -> LlamaTokenizer:
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    else:
        # Otherwise search for tokenizer
        tokenizer_path = find_tokenizer(model_path)

    return LlamaTokenizer.from_pretrained(
        str(tokenizer_path.parent),
        local_files_only=True,
        tokenizer_file=str(tokenizer_path.name),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", choices=["7B", "13B"], default="7B")
    parser.add_argument("--input", default="This is an example prompt.")
    parser.add_argument("--tokenizer", help="Optional explicit tokenizer path")
    args = parser.parse_args()

    model_config = {
        "7B": {"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
        "13B": {"num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
    }[args.size]

    profiler = Profiler(gpu=True)

    try:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        tokenizer = load_tokenizer(args)
        profiler.on_stage("tokenizer_loaded")

        te.initialize()

        has_gpu = torch.cuda.is_available()
        device = torch.device("cuda" if has_gpu else "cpu")

        # Megatron's tensor parallel
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,  # No tensor parallelism for now
            pipeline_model_parallel_size=1,  # No pipeline parallelism
        )
        profiler.on_stage("megatron_initialized")

        # Megatron transformer config
        config = TransformerConfig(
            num_layers=model_config["num_layers"],
            hidden_size=model_config["hidden_size"],
            num_attention_heads=model_config["num_attention_heads"],
            max_position_embeddings=4096,
            init_method_std=0.02,
            use_scaled_init_method=True,
            attention_softmax_in_fp32=True,
            rotary_pct=0.25,  # LLaMA uses rotary embeddings
        )

        model = GPTModel(  # noqa: F821
            config=config,
            vocab_size=tokenizer.vocab_size,
            max_sequence_length=4096,
            parallel_output=False,
            share_embeddings_and_output_weights=True,
        )
        if has_gpu:
            model = model.cuda()

        profiler.on_stage("model_created")

        # Load DMC checkpoint directly to GPU
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        profiler.on_stage("model_loaded")

        # Run inference
        input_ids = tokenizer(args.input, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids["input_ids"])
            logits = output.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([input_ids["input_ids"], next_token.unsqueeze(-1)], dim=-1)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            profiler.track("output", text)
            profiler.on_stage("inference_complete")

    except Exception as e:
        profiler.track_error("megatron", str(e))
