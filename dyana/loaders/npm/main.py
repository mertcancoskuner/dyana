import argparse
import json
import subprocess

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install a NodeJS package via NPM")
    parser.add_argument("--package", help="NPM compatible package name or expression", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    try:
        subprocess.check_call(["npm", "install", args.package])
        profiler.track_memory("after_installation")
        profiler.track_disk("after_installation")
    except Exception as e:
        profiler.track_error("npm", str(e))

    print(json.dumps(profiler.as_dict()))
