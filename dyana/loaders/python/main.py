import argparse
import os
import runpy

from dyana import Profiler, capture_output  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Python file")
    parser.add_argument("--script", help="Path to Python file", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    if not os.path.exists(args.script):
        profiler.track_error("python", "Python file not found")
    else:
        try:
            with capture_output() as (stdout_buffer, stderr_buffer):
                runpy.run_path(args.script)

                profiler.on_stage("after_execution")
                profiler.track("stdout", stdout_buffer.getvalue())
                profiler.track("stderr", stderr_buffer.getvalue())
        except Exception as e:
            profiler.track_error("python", str(e))
