<<<<<<< Updated upstream
import os
import sys
=======
import argparse
import os
>>>>>>> Stashed changes
import torch
from pathlib import Path
from dyana import Profiler

<<<<<<< Updated upstream
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training import get_args, get_model
from megatron.training.arguments import parse_args, core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.contrib.dmc import add_dmc_layer


def setup_megatron_args(model_size: str, model_path: str, tokenizer_path: str):
    """Setup Megatron arguments"""
    print("Debug: Starting argument setup")
    sys.argv = [sys.argv[0]]

    args = [
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--load",
        model_path,
        "--tokenizer-model",
        tokenizer_path,
        "--tokenizer-type",
        "Llama2Tokenizer",
        "--bf16",
        "--seq-length",
        "4096",
        "--max-position-embeddings",
        "4096",
        "--num-layers",
        "32" if model_size == "7B" else "40",
        "--hidden-size",
        "4096" if model_size == "7B" else "5120",
        "--num-attention-heads",
        "32" if model_size == "7B" else "40",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--no-masked-softmax-fusion",
        "--no-load-optim",
        "--no-load-rng",
        "--skip-train",
        "--fp16",
        "--use-cpu-initialization",  # avoid CUDA deadlocks
        "--tokenizer-type",
        "Llama2Tokenizer",
    ]

    print("Debug: Setting sys.argv")
    sys.argv.extend(args)

    print("Debug: Parsing args")
    args = parse_args()

    print("Debug: Initializing Megatron")
    initialize_megatron(args_defaults={"no_load_optim": True, "no_load_rng": True})

    return get_args()


def model_provider(pre_process=True, post_process=True):
    """Model provider for Megatron to load the model."""
    print("Debug: Setting up model provider")
    args = get_args()
    config = core_transformer_config_from_args(args)

    print("Debug: Creating model")
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--size", choices=["7B", "13B"], required=True)
    parser.add_argument("--input", default="This is an example prompt.")
    args = parser.parse_args()

    profiler = Profiler(gpu=True)

    try:
        # verify files
        model_path = Path(args.model)
        tokenizer_path = Path(args.tokenizer)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        print("Debug: Starting initialization")
        profiler.on_stage("initializing")

        print("Debug: Setting up args")
        args = setup_megatron_args(args.size, str(model_path), str(tokenizer_path))

        print("Debug: Initializing model parallel")
        torch.cuda.empty_cache()
        parallel_state.set_tensor_model_parallel_world_size(1)
        parallel_state.set_tensor_model_parallel_rank(0)

        print("Debug: Creating model")
        model = get_model(model_provider, wrap_with_ddp=False)

        print("Debug: Loading checkpoint")
        _ = load_checkpoint(model[0], None, None)
        model = model[0].cuda()
        model.eval()

        print("Loading tokenizer...")
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path))

        print("Starting inference...")
        input_ids = tokenizer(args.input, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(input_ids=input_ids["input_ids"], max_new_tokens=100, use_cache=True)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            profiler.track("output", text)
            print(f"Generated text: {text}")

        profiler.on_stage("complete")

    except Exception as e:
        print(f"Debug: Error occurred: {str(e)}")
        print(f"Debug: Error type: {type(e)}")
        import traceback

        print(f"Debug: Traceback: {traceback.format_exc()}")
        profiler.track_error("model", str(e))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
=======

def verify_cuda_setup():
    """Verify CUDA and PyTorch setup before model loading"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Disable JIT/Inductor
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)

    print("=== Runtime Configuration ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print("===========================")

    # Set default device
    torch.cuda.set_device(0)


if __name__ == "__main__":
    # Initialize profiler first
    profiler = Profiler(gpu=True)

    try:
        # Verify CUDA setup
        verify_cuda_setup()
        profiler.on_stage("cuda_verified")

        os.environ["TE_VERBOSE"] = "1"
        os.environ["NVTE_FRAMEWORK"] = "pytorch"
        print("Starting Megatron loader with verbose logging...")

        # initialize CUDA and Transformer
        if torch.cuda.is_available():
            import transformer_engine.pytorch as te

            te.initialize()
            print(f"Initialized Transformer Engine version: {te.__version__}")

        # import Megatron dependencies
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

            model_path = Path(args.model)
            tokenizer_path = Path(args.tokenizer)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            # Initialize Megatron's tensor parallel
            world_size = torch.cuda.device_count()
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
            profiler.on_stage("megatron_initialized")

            # Model config based on size
            model_config = {
                "7B": {"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32},
                "13B": {"num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40},
            }[args.size]

            # Create Megatron transformer config
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
>>>>>>> Stashed changes
