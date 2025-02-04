import argparse
import os
from pathlib import Path

import torch
import transformer_engine as te
from megatron.model.gpt_model import GPTModel

from dyana.profiler import Profiler


def verify_cuda_setup() -> None:
    """Verify CUDA and PyTorch setup before model loading"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Disable JIT/Inductor features
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)

    print("=== Runtime Configuration ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("===========================")

    torch.cuda.set_device(0)


if __name__ == "__main__":
    profiler = Profiler(gpu=True)

    try:
        # Verify CUDA setup
        verify_cuda_setup()
        profiler.on_stage("cuda_verified")

        os.environ["TE_VERBOSE"] = "1"
        os.environ["NVTE_FRAMEWORK"] = "pytorch"
        print("Starting Megatron loader with verbose logging...")

        # Initialize CUDA and Transformer Engine
        if torch.cuda.is_available():
            import transformer_engine.pytorch as te

            te.initialize()
            print(f"Initialized Transformer Engine version: {te.__version__}")

        from megatron.core import parallel_state
        from megatron.core.transformer.transformer_config import TransformerConfig
        from transformers import LlamaTokenizer

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True)
        parser.add_argument("--tokenizer", required=True)
        parser.add_argument("--size", choices=["7B", "13B"], required=True)
        parser.add_argument("--input", default="This is an example prompt.")
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            print(f"Transformer Engine version: {te.__version__}")
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"CUDA version: {torch.version.cuda}")
            profiler.track(
                "env_info",
                {
                    "te_version": te.__version__,
                    "cuda_devices": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                },
            )

            model_path = Path(args.model)
            tokenizer_path = Path(args.tokenizer)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            # Initialize Megatron's tensor parallel
            world_size = torch.cuda.device_count()
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,  # No tensor parallelism for now
                pipeline_model_parallel_size=1,  # No pipeline parallelism
            )
            profiler.on_stage("megatron_initialized")

            # Model config based on size
            model_config = {
                "7B": {"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
                "13B": {"num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
            }[args.size]

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

                checkpoint = torch.load(str(model_path), map_location=device)
                model.load_state_dict(checkpoint)
                model.cuda()
                model.eval()
                profiler.on_stage("model_loaded")

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
            parallel_state.destroy_model_parallel()
        except Exception as e:
            profiler.track_error("cleanup", str(e))
            print(f"Cleanup error: {e}")
