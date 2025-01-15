import argparse
import json
import os
import subprocess
import typing as t

from dyana import get_peak_rss  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ELF file")
    parser.add_argument("--elf", help="Path to ELF file", required=True)
    args = parser.parse_args()

    result: dict[str, t.Any] = {
        "ram": {"start": get_peak_rss()},
        "errors": {},
        "stdout": None,
        "stderr": None,
        "exit_code": None,
    }

    if not os.path.exists(args.elf):
        result["errors"]["elf"] = "ELF file not found"
    else:
        try:
            ret = subprocess.run([args.elf], capture_output=True, text=True, errors="replace")

            result["ram"]["after_execution"] = get_peak_rss()

            result["stdout"] = ret.stdout
            result["stderr"] = ret.stderr
            result["exit_code"] = ret.returncode
        except Exception as e:
            result["errors"]["elf"] = str(e)

    print(json.dumps(result))
