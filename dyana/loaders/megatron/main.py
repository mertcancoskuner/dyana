# ruff: noqa: I001, F401, E402, B904, F821
# type: ignore
import os
import sys
import logging
import warnings
import argparse
from pathlib import Path
from io import StringIO
import contextlib

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Import torch and configure CUDA
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.init()
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

    print("\n=== Tokenizer Search ===", file=sys.stderr)

    for directory in search_dirs:
        print(f"Looking in: {directory}", file=sys.stderr)
        print("Directory contents:", file=sys.stderr)
        all_files = list(directory.glob("*"))
        for f in sorted(all_files):
            print(f"  {f}", file=sys.stderr)
            # If it looks like a LLaMA tokenizer file, try it first
            if "tokenizer" in f.name.lower() and f.name.endswith(".model"):
                print(f"Found likely LLaMA tokenizer: {f}", file=sys.stderr)
                return f

        # If no obvious tokenizer found, try the patterns
        print("\nTrying patterns:", file=sys.stderr)
        for pattern in patterns:
            print(f"  {pattern}...", file=sys.stderr, end=" ")
            matches = list(directory.glob(pattern))
            if matches:
                print(f"Found: {matches[0]}", file=sys.stderr)
                return matches[0]
            print("No match", file=sys.stderr)

    raise FileNotFoundError(
        f"No tokenizer found in {[str(d) for d in search_dirs]} after trying patterns: {patterns}\n"
        f"Available files in {model_path.parent}: {[f.name for f in model_path.parent.glob('*')]}"
    )


if __name__ == "__main__":
    # Set multiprocessing start method
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
        try:
            print("=== Starting Megatron Loader ===", file=sys.stderr)
            from dyana import Profiler

            # Initialize CUDA
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "0"
            os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
            os.environ["TORCH_INDUCTOR_DISABLE_CUDA_GRAPH"] = "1"  # Disable CUDA graphs

            if not os.path.exists("/dev/shm"):
                print("Warning: /dev/shm not found, creating...", file=sys.stderr)
                os.makedirs("/dev/shm", exist_ok=True)

            profiler = Profiler(gpu=True)

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but required")

            # Force CUDA initialization
            torch.cuda.init()
            torch.cuda.set_device(0)
            # Allocate a small tensor to ensure CUDA is working
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()

            # GPU info
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            print(
                f"Found {device_count} CUDA devices, using {device_name} with {gpu_mem / 1e9:.1f}GB memory",
                file=sys.stderr,
            )
            profiler.track(
                "gpu_info", {"device": device_name, "count": device_count, "cuda": cuda_version, "memory": gpu_mem}
            )
            profiler.on_stage("cuda_initialized")

            print("\n=== Importing Dependencies ===", file=sys.stderr)
            try:
                from transformers import LlamaTokenizer

                print("✓ Imported LlamaTokenizer", file=sys.stderr)
                from megatron.core import parallel_state

                print("✓ Imported parallel_state", file=sys.stderr)
                from megatron.core.transformer.transformer_config import TransformerConfig

                print("✓ Imported TransformerConfig", file=sys.stderr)
            except Exception as e:
                print(f"Failed to import dependencies: {e}", file=sys.stderr)
                profiler.track_error("imports", str(e))
                raise

            print("\n=== Parsing Arguments ===", file=sys.stderr)
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", required=True)
            parser.add_argument("--size", choices=["7B", "13B"], required=True)
            parser.add_argument("--input", default="This is an example prompt.")
            parser.add_argument("--tokenizer", help="Optional explicit tokenizer path")
            args = parser.parse_args()

            model_path = Path(args.model)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            print("\n=== Checking Files ===", file=sys.stderr)
            print(f"Model path: {model_path}", file=sys.stderr)
            print("Directory contents:", file=sys.stderr)
            for f in sorted(model_path.parent.glob("*")):
                print(f"  {f}", file=sys.stderr)

            # Try explicit tokenizer path
            if args.tokenizer:
                tokenizer_path = Path(args.tokenizer)
                if not tokenizer_path.exists():
                    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
                print(f"Using provided tokenizer: {tokenizer_path}", file=sys.stderr)
            else:
                # Otherwise search for tokenizer
                tokenizer_path = find_tokenizer(model_path)
                print(f"Found tokenizer: {tokenizer_path}", file=sys.stderr)

            try:
                print("\n=== Loading Tokenizer ===", file=sys.stderr)
                print(f"Loading from: {tokenizer_path}", file=sys.stderr)

                try:
                    tokenizer = LlamaTokenizer.from_pretrained(
                        str(tokenizer_path.parent),
                        local_files_only=True,
                        tokenizer_file=str(tokenizer_path.name),
                    )
                    print(f"Successfully loaded tokenizer (vocab_size={tokenizer.vocab_size})", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to load tokenizer from {tokenizer_path}: {e}", file=sys.stderr)
                    raise
                print("=======================\n", file=sys.stderr)
                profiler.on_stage("tokenizer_loaded")
            except Exception as e:
                print(f"Error loading tokenizer: {e}", file=sys.stderr)
                profiler.track_error("tokenizer", str(e))
                raise

            # Initialize profiler first
            initialized_parallel = False

            try:
                # Use fork multiprocessing
                if sys.platform == "linux":
                    import torch.multiprocessing as mp

                    mp.set_start_method("fork", force=True)

                if torch.cuda.is_available():
                    print("=== Runtime Configuration ===")
                    print(f"PyTorch: {torch.__version__}")
                    print(f"CUDA: {torch.version.cuda}")
                    print(f"Device: {torch.cuda.get_device_name()}")
                    print("===========================")
                    profiler.on_stage("cuda_verified")

                if torch.cuda.is_available():
                    import transformer_engine.pytorch as te

                    try:
                        te.initialize()
                        print(f"Initialized Transformer Engine version: {te.__version__}")
                    except Exception as e:
                        print(f"Warning: Transformer Engine initialization failed: {e}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                try:
                    print(f"Transformer Engine version: {te.__version__}")  # noqa: F821
                    print(f"CUDA devices: {torch.cuda.device_count()}")
                    print(f"CUDA version: {torch.version.cuda}")
                    profiler.track(
                        "env_info",
                        {
                            "te_version": te.__version__,  # noqa: F821
                            "cuda_devices": torch.cuda.device_count(),
                            "cuda_version": torch.version.cuda,
                        },
                    )

                    # Megatron's tensor parallel
                    world_size = torch.cuda.device_count()
                    parallel_state.initialize_model_parallel(
                        tensor_model_parallel_size=1,  # No tensor parallelism for now
                        pipeline_model_parallel_size=1,  # No pipeline parallelism
                    )
                    profiler.on_stage("megatron_initialized")

                    # parallel state initialization
                    initialized_parallel = True

                    # Model config
                    model_config = {
                        "7B": {"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
                        "13B": {"num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
                    }[args.size]

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
                    profiler.track("model_config", model_config)
                    profiler.on_stage("config_created")

                    try:
                        # Load tokenizer
                        print("\n=== Loading Tokenizer ===", file=sys.stderr)
                        print(f"Loading from: {tokenizer_path}", file=sys.stderr)
                        tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path.parent), local_files_only=True)
                        print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}", file=sys.stderr)
                        print("=======================\n", file=sys.stderr)
                        profiler.on_stage("tokenizer_loaded")

                        model = GPTModel(  # noqa: F821
                            config=config,
                            vocab_size=tokenizer.vocab_size,
                            max_sequence_length=4096,
                            parallel_output=False,
                            share_embeddings_and_output_weights=True,
                        ).cuda()  # Explicit GPU
                        profiler.on_stage("model_created")

                        # Load DMC checkpoint directly to GPU
                        checkpoint = torch.load(str(model_path), map_location="cuda")
                        model.load_state_dict(checkpoint)
                        model.eval()
                        torch.cuda.synchronize()  # Ensure model is loaded to GPU
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
                        profiler.track_error("model", str(e))
                        print(f"Model loading/inference failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise

                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    profiler.track_error("model", str(e))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

            except Exception as e:
                profiler.track_error("setup", str(e))
                print(f"Setup error: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

            finally:
                try:
                    if initialized_parallel:
                        parallel_state.destroy_model_parallel()
                except Exception as e:
                    profiler.track_error("cleanup", str(e))
                    print(f"Cleanup error: {e}")

        except Exception as e:
            profiler.track_error("runtime", str(e))
            print(f"Error: {e}", file=sys.stderr)
            raise
        finally:
            profiler.flush()
            print(captured_output.getvalue(), file=sys.stderr)
