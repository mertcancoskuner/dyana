import argparse
import json
import os
import resource
import subprocess
import typing as t


def get_peak_rss() -> int:
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Python file")
    parser.add_argument("--script", help="Path to Python file", required=True)
    args = parser.parse_args()

    result: dict[str, t.Any] = {
        "ram": {"start": get_peak_rss()},
        "errors": {},
        "stdout": None,
        "stderr": None,
        "exit_code": None,
    }

    if not os.path.exists(args.script):
        result["errors"]["elf"] = "Python file not found"
    else:
        try:
            ret = subprocess.run(["python", args.script], capture_output=True, text=True, errors="replace")

            result["ram"]["after_execution"] = get_peak_rss()

            result["stdout"] = ret.stdout
            result["stderr"] = ret.stderr
            result["exit_code"] = ret.returncode
        except Exception as e:
            result["errors"]["elf"] = str(e)

    print(json.dumps(result))
