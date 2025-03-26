import argparse
import os
import pickle
import subprocess
import sys

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Python pickle file")
    parser.add_argument("--pickle", help="Path to pickle file", required=True)
    parser.add_argument("--extra-requirements", help="Extra pip requirements", default="")
    args = parser.parse_args()
    profiler: Profiler = Profiler(gpu=True)

    # Install any extra dependencies requested
    if args.extra_requirements:
        try:
            print(f"Installing runtime dependencies: {args.extra_requirements}")
            requirements = args.extra_requirements.split(",")
            for req in requirements:
                req = req.strip()
                if req:
                    print(f"Installing dependency: {req}")
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--no-cache-dir", req], capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        profiler.track_warning("dependencies", f"Failed to install {req}: {result.stderr}")
                        print(f"Warning: Failed to install {req}: {result.stderr}")
                    else:
                        print(f"Successfully installed {req}")
        except Exception as e:
            profiler.track_error("dependencies", f"Failed to install dependencies: {str(e)}")
            print(f"Error installing dependencies: {str(e)}")

    if not os.path.exists(args.pickle):
        profiler.track_error("pickle", "Pickle file not found")
    else:
        try:
            with open(args.pickle, "rb") as f:
                data = pickle.load(f)
                profiler.on_stage("after_load")

                # Try to get attributes of the loaded object
                if hasattr(data, "__dict__"):
                    profiler.track_extra("object_attributes", list(data.__dict__.keys()))

                # Try to get the type
                profiler.track_extra("object_type", str(type(data)))

                # Try to get the shape for numpy arrays or tensors
                if hasattr(data, "shape"):
                    profiler.track_extra("shape", str(data.shape))

                # Try to get the length for lists, tuples, dicts
                if hasattr(data, "__len__"):
                    profiler.track_extra("length", len(data))
        except ImportError as e:
            profiler.track_error("pickle", str(e))
        except Exception as e:
            profiler.track_error("pickle", str(e))
