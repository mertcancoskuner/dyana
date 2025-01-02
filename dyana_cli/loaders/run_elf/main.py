import argparse
import json
import resource
import os


def get_peak_rss() -> int:
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ELF file")
    parser.add_argument("--elf", help="Path to ELF file", required=True)
    args = parser.parse_args()

    stdout: str = ""
    stderr: str = ""
    errors: dict[str, str | None] = {}
    ram: dict[str, int] = {"start": get_peak_rss()}

    if not os.path.exists(args.elf):
        errors["elf"] = "ELF file not found"
    else:
        os.system(f"{args.elf} > /dev/null 2>&1")

    print(
        json.dumps(
            {
                "ram": ram,
                "errors": errors,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    )
