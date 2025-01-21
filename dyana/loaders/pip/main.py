import argparse
import importlib
import json
import re
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

        # explicitly require the package to make sure it's loaded
        package_name = re.split("[^a-zA-Z0-9_-]", args.package)[0]
        # normalize
        normalized_package_name = package_name.strip().lower().replace("-", "_")
        try:
            importlib.import_module(normalized_package_name)
        except Exception as e:
            profiler.track("stderr", str(e))
    except Exception as e:
        profiler.track_error("pip", str(e))

    print(json.dumps(profiler.as_dict()))
