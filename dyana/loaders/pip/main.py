import argparse
import glob
import importlib
import json
import os
import re
import subprocess
import sys
from typing import Any

from dyana import Profiler  # type: ignore[attr-defined]


def find_site_packages() -> str | None:
    """Find the site-packages directory where pip installs packages."""
    for path in sys.path:
        if path.endswith("site-packages"):
            return path
    return None


def debug_package_metadata(package_name: str, profiler: Profiler) -> None:
    """Debug helper to inspect package metadata files."""
    site_packages = find_site_packages()
    if not site_packages:
        profiler.track("debug", "Could not find site-packages directory")
        return

    # list related files
    dist_info = list(glob.glob(os.path.join(site_packages, f"{package_name}*.dist-info")))
    egg_info = list(glob.glob(os.path.join(site_packages, f"{package_name}*.egg-info")))
    package_files = list(glob.glob(os.path.join(site_packages, f"{package_name}*")))

    debug_info: dict[str, Any] = {
        "site_packages": site_packages,
        "dist_info_found": dist_info,
        "egg_info_found": egg_info,
        "package_files": package_files,
        "top_level_contents": dict[str, str](),
    }

    # check top_level.txt
    for d in dist_info + egg_info:
        top_level = os.path.join(d, "top_level.txt")
        if os.path.exists(top_level):
            with open(top_level) as f:
                debug_info["top_level_contents"][d] = f.read().strip()

    profiler.track("debug", debug_info)


def get_package_import_names(package_name: str) -> list[str]:
    """Get possible import names for a package using various methods."""
    site_packages = find_site_packages()
    if not site_packages:
        return []

    import_names = set()

    # look for package name variations
    base_name = package_name.replace("-", "_")
    variations = [
        package_name,
        base_name,
        base_name.lower(),
    ]

    # filter out standard library modules
    stdlib_modules = sys.stdlib_module_names

    for variant in variations:
        # Only look in site-packages directory
        package_path = os.path.join(site_packages, variant)
        if os.path.exists(package_path):
            if os.path.isfile(package_path + ".py"):
                import_names.add(variant)
            elif os.path.isdir(package_path) and os.path.exists(os.path.join(package_path, "__init__.py")):
                import_names.add(variant)

        # check dist-info directory for this specific variant
        dist_info_pattern = os.path.join(site_packages, f"{variant}*.dist-info")
        for dist_info_dir in glob.glob(dist_info_pattern):
            # try top_level.txt
            top_level = os.path.join(dist_info_dir, "top_level.txt")
            if os.path.exists(top_level):
                with open(top_level) as f:
                    for name in f.readlines():
                        name = name.strip()
                        if name and name not in stdlib_modules:
                            import_names.add(name)

    return [name for name in import_names if name not in stdlib_modules]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install a Python package via PIP")
    parser.add_argument("--package", help="PIP compatible package name or expression", required=True)
    args = parser.parse_args()
    profiler: Profiler = Profiler()

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--root-user-action=ignore", args.package]
        )
        profiler.track_memory("after_installation")
        profiler.track_disk("after_installation")

        importlib.invalidate_caches()
        site_packages = find_site_packages()
        if site_packages and site_packages not in sys.path:
            sys.path.append(site_packages)

        # get base package name (remove version, etc)
        package_name = re.split("[^a-zA-Z0-9_-]", args.package)[0]

        debug_package_metadata(package_name, profiler)

        import_success = False
        import_errors = []
        successful_name = None

        import_names = get_package_import_names(package_name)

        print(f"\nAttempting imports with names: {import_names}\n")

        if import_names:
            import_names.sort(key=len)

            for name in import_names:
                try:
                    importlib.import_module(name)
                    import_success = True
                    successful_name = name
                    print(f"\nSuccessfully imported as '{name}'\n")
                    break
                except ImportError as e:
                    import_errors.append(f"Import name '{name}': {str(e)}")

        if not import_success:
            normalized_name = package_name.strip().lower().replace("-", "_")
            try:
                importlib.import_module(normalized_name)
                import_success = True
                successful_name = normalized_name
                print(f"\nSuccessfully imported using normalized name '{normalized_name}'\n")
            except ImportError as e:
                import_errors.append(f"Normalized name '{normalized_name}': {str(e)}")

        if import_success:
            profiler.track("result", f"Successfully imported as '{successful_name}'")
        else:
            profiler.track("stderr", "\n".join(import_errors))

    except Exception as e:
        profiler.track_error("pip", str(e))

    print(json.dumps(profiler.as_dict()))
