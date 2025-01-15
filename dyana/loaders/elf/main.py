import argparse
import json
import os
import subprocess

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ELF file")
    parser.add_argument("--elf", help="Path to ELF file", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    if not os.path.exists(args.elf):
        profiler.track_error("elf", "ELF file not found")
    else:
        try:
            ret = subprocess.run([args.elf], capture_output=True, text=True, errors="replace")

            profiler.track_memory("after_execution")

            profiler.track("stdout", ret.stdout)
            profiler.track("stderr", ret.stderr)
            profiler.track("exit_code", ret.returncode)
        except Exception as e:
            profiler.track_error("elf", str(e))

    print(json.dumps(profiler.as_dict()))
