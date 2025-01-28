import argparse
import os
import time

from ollama import Client

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Ollama model")
    parser.add_argument("--model", help="Name of the Ollama model to profile", required=True)
    parser.add_argument("--input", help="The input sentence", default="This is an example sentence.")
    args = parser.parse_args()

    # start ollama server
    os.system("ollama serve > /dev/null 2>&1 &")
    for _ in range(30):
        # print(f"waiting for ollama to start... {i}")
        if os.system("ollama ls > /dev/null 2>&1") == 0:
            break
        time.sleep(1)

    # create profiler after the server is started
    profiler: Profiler = Profiler(gpu=True)

    try:
        client = Client(
            host="http://127.0.0.1:11434",
        )
        response = client.chat(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": args.input,
                },
            ],
        )

        profiler.on_stage("after_inference")

        print(response)

    except Exception as e:
        profiler.track_error("ollama", str(e))
