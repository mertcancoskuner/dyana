import argparse
import json
import os
import pickle
import resource
import typing as t


def get_peak_rss() -> int:
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pickle file")
    parser.add_argument("--pickle", help="Path to pickle file", required=True)
    args = parser.parse_args()

    result: dict[str, t.Any] = {
        "ram": {"start": get_peak_rss()},
        "errors": {},
    }

    if not os.path.exists(args.pickle):
        result["errors"]["pickle"] = "pickle file not found"
    else:
        try:
            with open(args.pickle, "rb") as f:
                ret = pickle.load(f)
            result["ram"]["after_load"] = get_peak_rss()
        except Exception as e:
            result["errors"]["pickle"] = str(e)

    print(json.dumps(result))
