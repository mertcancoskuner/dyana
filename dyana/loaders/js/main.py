import argparse
import os
import subprocess

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Javascript file")
    parser.add_argument("--script", help="Path to Javascript file", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    if not os.path.exists(args.script):
        profiler.track_error("js", "Javascript file not found")
    else:
        try:
            result = subprocess.run(["node", args.script], capture_output=True, text=True)

            profiler.on_stage("after_execution")

            profiler.track("exit_code", result.returncode)
            profiler.track("stdout", result.stdout)
            profiler.track("stderr", result.stderr)
        except Exception as e:
            profiler.track_error("js", str(e))
