import argparse
import re
import subprocess

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install a NodeJS package via NPM")
    parser.add_argument("--package", help="NPM compatible package name or expression", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    try:
        subprocess.check_call(["npm", "install", args.package])
        profiler.on_stage("after_installation")

        # explicitly require the package to make sure it's loaded
        package_name = args.package
        if package_name.startswith("@"):
            package_name = package_name[1:]
        package_name = re.split("[^a-zA-Z0-9_-]", package_name)[0]
        result = subprocess.run(["node", "-e", f"require('{package_name}')"], capture_output=True, text=True)

        profiler.on_stage("after_require")

        profiler.track("exit_code", result.returncode)
        profiler.track("stdout", result.stdout)
        profiler.track("stderr", result.stderr)

    except Exception as e:
        profiler.track_error("npm", str(e))
