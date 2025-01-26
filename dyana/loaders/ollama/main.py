import argparse
import os
import subprocess
import time

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Ollama model")
    parser.add_argument("--model", help="Name of the Ollama model to profile", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    args = parser.parse_args()

    # start ollama server
    os.system("ollama serve > /dev/null 2>&1 &")
    for i in range(10):
        print(f"waiting for ollama to start... {i}")
        if os.system("ollama ls > /dev/null 2>&1") == 0:
            break
        time.sleep(1)

    # create profiler after the server is started
    profiler: Profiler = Profiler()

    try:
        result = subprocess.run(["ollama", "run", args.model, args.input], capture_output=True, text=True)

        profiler.track_memory("after_run")
        profiler.track("exit_code", result.returncode)
        profiler.track("stdout", result.stdout)
        profiler.track("stderr", result.stderr)
    except Exception as e:
        profiler.track_error("ollama", str(e))
