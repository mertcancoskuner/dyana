import argparse
import json
import subprocess
import sys

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install a Python package via PIP")
    parser.add_argument("--package", help="PIP compatible package name or expression", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--root-user-action=ignore", args.package]
        )
        profiler.track_memory("after_installation")
        profiler.track_disk("after_installation")
    except Exception as e:
        profiler.track_error("pip", str(e))

    print(json.dumps(profiler.as_dict()))
