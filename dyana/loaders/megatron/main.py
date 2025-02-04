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
os.environ["PYTHONWARNINGS"] = "ignore"

os.environ.update(
    {
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "TORCH_USE_CUDA_DSA": "0",
        "NVTE_FRAMEWORK": "pytorch",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32",
        "TORCH_INDUCTOR_DISABLE_CUDA_GRAPH": "1",
        "TORCH_INDUCTOR_USE_PYTHON_BINDING": "0",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    }
)

import torch

torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)

if __name__ == "__main__":
    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
        try:
            from dyana import Profiler

            profiler = Profiler(gpu=True)

            # Initialize CUDA
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_device(0)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                profiler.track(
                    "cuda_info",
                    {
                        "version": torch.version.cuda,
                        "device": torch.cuda.get_device_name(),
                        "device_count": torch.cuda.device_count(),
                    },
                )
            profiler.on_stage("cuda_initialized")

            parser = argparse.ArgumentParser()
            parser.add_argument("--model", required=True)
            parser.add_argument("--tokenizer", required=True)
            parser.add_argument("--size", choices=["7B", "13B"], required=True)
            parser.add_argument("--input", default="This is an example prompt.")
            args = parser.parse_args()

            model_path = Path(args.model)
            tokenizer_path = Path(args.tokenizer)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            profiler.on_stage("args_verified")

            from transformers import LlamaTokenizer
            from megatron.core import parallel_state
            from megatron.core.transformer.transformer_config import TransformerConfig

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
                    print(f"Transformer Engine version: {transformer_engine.__version__}")
                    print(f"CUDA devices: {torch.cuda.device_count()}")
                    print(f"CUDA version: {torch.version.cuda}")
                    profiler.track(
                        "env_info",
                        {
                            "te_version": transformer_engine.__version__,
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
                        tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path.parent), local_files_only=True)
                        profiler.on_stage("tokenizer_loaded")

                        model = GPTModel(
                            config=config,
                            vocab_size=tokenizer.vocab_size,
                            max_sequence_length=4096,
                            parallel_output=False,
                            share_embeddings_and_output_weights=True,
                        )
                        profiler.on_stage("model_created")

                        # Load DMC checkpoint
                        checkpoint = torch.load(str(model_path), map_location=device)
                        model.load_state_dict(checkpoint)
                        model.cuda()
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
                # Clean up Megatron's parallel state only if it was initialized
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
