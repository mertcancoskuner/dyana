import argparse
import json
import os
import sys
import time
import traceback
from typing import Any

from ollama import Client


def ensure_output(data: str) -> None:
    """Write to both stdout and stderr to ensure output is captured"""
    print(data)
    print(data, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()


try:
    ensure_output(json.dumps({"status": "script_started"}))

    try:
        from dyana import Profiler  # type: ignore[attr-defined]
    except ImportError:
        # Only define our own Profiler if the import fails
        class Profiler:  # type: ignore
            def __init__(self, gpu: bool = False) -> None:
                self.gpu = gpu

            def on_stage(self, stage: str) -> None:
                pass

            def track_error(self, source: str, error: str) -> None:
                pass

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run an Ollama model")
        parser.add_argument("--model", help="Name of the Ollama model to profile", required=True)
        parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
        args = parser.parse_args()

        result: dict[str, Any] = {
            "status": "started",
            "model": args.model,
            "input": args.input,
            "timestamp": time.time(),
        }
        ensure_output(json.dumps(result))

        try:
            # Create profiler
            profiler = Profiler(gpu=True)

            os.makedirs("/root/.ollama/manifests", exist_ok=True)
            os.makedirs("/root/.ollama/cache", exist_ok=True)

            try:
                os.chmod("/root/.ollama", 0o755)
                os.chmod("/root/.ollama/models", 0o755)
                os.chmod("/root/.ollama/manifests", 0o755)
                os.chmod("/root/.ollama/cache", 0o755)
            except Exception as perm_error:
                result["permission_warning"] = str(perm_error)

            # Start ollama server
            os.system("ollama serve > /dev/null 2>&1 &")

            # Wait for server to start
            server_started = False
            for i in range(30):
                if os.system("ollama ls > /dev/null 2>&1") == 0:
                    server_started = True
                    result["startup_time"] = i
                    break
                time.sleep(1)

            if not server_started:
                result["status"] = "error"
                result["error"] = "Failed to start Ollama server after 30 seconds"
                ensure_output(json.dumps(result))
                sys.exit(1)

            # Record initialization stage
            profiler.on_stage("initialization")

            # Connect to the ollama server
            client = Client(host="http://127.0.0.1:11434")

            # Check if model exists locally without trying to pull it first
            models = client.list()
            model_exists = any(m.get("name", "") == args.model for m in models.get("models", []))

            result["model_found"] = model_exists

            if model_exists:
                # Skip pulling if the model already exists
                result["status"] = "running_inference"
                ensure_output(json.dumps(result))

                # Run inference with existing model
                chat_response = client.chat(
                    model=args.model,
                    messages=[{"role": "user", "content": args.input}],
                )

                # Mark completion of inference
                profiler.on_stage("after_inference")

                # Update result with success
                result["status"] = "success"
                if hasattr(chat_response, "model_dump"):
                    result["response"] = chat_response.model_dump()
                else:
                    result["response"] = str(chat_response)
            else:
                # Can't pull models due to read-only filesystem
                result["status"] = "error"
                result["error"] = (
                    "Model not found locally and cannot pull due to read-only filesystem. Please pull the model on your host with 'ollama pull "
                    + args.model
                    + "' before running dyana."
                )
                ensure_output(json.dumps(result))

        except Exception as e:
            # Handle any exceptions
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            if "profiler" in locals():
                profiler.track_error("ollama", str(e))

        # Output final result
        ensure_output(json.dumps(result, default=str))

except Exception as outer_e:
    # Last resort error handling
    emergency_data: dict[str, str] = {
        "status": "fatal_error",
        "error": str(outer_e),
        "traceback": traceback.format_exc(),
    }
    ensure_output(json.dumps(emergency_data))
