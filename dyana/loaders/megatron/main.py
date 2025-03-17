# Standard library imports
import argparse
import contextlib
import logging
import multiprocessing
import sys
import warnings
from io import StringIO
from pathlib import Path

# Third-party imports
import torch
from te.utils import te

try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.model import GPTModel
except ImportError:
    # For type checking only - these will never run
    print("Warning: Megatron modules not available, using stubs for type checking", file=sys.stderr)

from transformers import LlamaTokenizer

# Local imports
from dyana import Profiler  # type: ignore[attr-defined]


def safe_cuda_init() -> None:
    """Initialize CUDA with proper type annotations."""
    if hasattr(torch.cuda, "init"):
        torch.cuda.init()  # type: ignore[no-untyped-call]


logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    safe_cuda_init()
    torch.cuda.set_device(0)


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


def load_tokenizer(args: argparse.Namespace) -> LlamaTokenizer:
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
    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer file")
    parser.add_argument("--input", type=str, default="Hello, world!", help="Input text to process")
    parser.add_argument("--model-config", type=str, help="Path to model config JSON file")
    args = parser.parse_args()

    profiler = Profiler(gpu=torch.cuda.is_available())

    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
        try:
            print("=== Starting Megatron Loader ===", file=sys.stderr)
        except Exception as e:
            print(f"Error during output capture: {e}", file=sys.stderr)

    try:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        tokenizer = load_tokenizer(args)
        profiler.on_stage("tokenizer_loaded")

        try:
            te.initialize()
        except NameError:
            pass

        has_gpu = torch.cuda.is_available()
        device = torch.device("cuda" if has_gpu else "cpu")

        if has_gpu:
            # Force CUDA initialization
            safe_cuda_init()
            torch.cuda.set_device(0)
            # Allocate a small tensor to ensure CUDA is working
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()

        model_config = {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
        }  # Default values

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

        model = GPTModel(
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

    profiler.flush()
