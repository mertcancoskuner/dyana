import argparse
import json
import os
import pickle
import typing as t

from dyana import get_current_imports, get_peak_rss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pickle file")
    parser.add_argument("--pickle", help="Path to pickle file", required=True)
    args = parser.parse_args()

    result: dict[str, t.Any] = {
        "ram": {"start": get_peak_rss()},
        "errors": {},
    }

    imports_at_start = get_current_imports()

    if not os.path.exists(args.pickle):
        result["errors"]["pickle"] = "pickle file not found"
    else:
        try:
            with open(args.pickle, "rb") as f:
                ret = pickle.load(f)
            result["ram"]["after_load"] = get_peak_rss()
        except Exception as e:
            result["errors"]["pickle"] = str(e)

    imports_at_end = get_current_imports()
    result["extra"] = {"imports": {k: imports_at_end[k] for k in imports_at_end if k not in imports_at_start}}

    print(json.dumps(result))
