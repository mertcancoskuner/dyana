import argparse
import logging
import os

from peft import load_peft_weights

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    logging.disable(logging.ERROR)

    parser = argparse.ArgumentParser(description="Profile LoRA adapter files")
    parser.add_argument("--adapter", help="Path to LoRA adapter", required=True)
    args = parser.parse_args()

    path: str = os.path.abspath(args.adapter)
    profiler: Profiler = Profiler(gpu=True)

    try:
        load_peft_weights(path)
        profiler.track_memory("after_adapter_loaded")
    except Exception as e:
        profiler.track_error("adapter", str(e))
