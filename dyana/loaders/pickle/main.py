import argparse
import json
import os
import pickle

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pickle file")
    parser.add_argument("--pickle", help="Path to pickle file", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    if not os.path.exists(args.pickle):
        profiler.track_error("pickle", "pickle file not found")
    else:
        try:
            with open(args.pickle, "rb") as f:
                ret = pickle.load(f)
            profiler.track_memory("after_load")
        except Exception as e:
            profiler.track_error("pickle", str(e))
