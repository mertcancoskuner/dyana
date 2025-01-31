import os
import sys
import torch
from pathlib import Path
from dyana import Profiler

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
