import argparse
import json
import os
import runpy
import typing as t

from dyana import capture_output, get_current_imports, get_peak_rss

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

    imports_at_start = get_current_imports()

    if not os.path.exists(args.script):
        result["errors"]["elf"] = "Python file not found"
    else:
        try:
            with capture_output() as (stdout_buffer, stderr_buffer):
                runpy.run_path(args.script)

                result["ram"]["after_execution"] = get_peak_rss()
                result["stdout"] = stdout_buffer.getvalue()
                result["stderr"] = stderr_buffer.getvalue()
        except Exception as e:
            result["errors"]["elf"] = str(e)

    imports_at_end = get_current_imports()
    result["extra"] = {"imports": {k: imports_at_end[k] for k in imports_at_end if k not in imports_at_start}}

    print(json.dumps(result))
