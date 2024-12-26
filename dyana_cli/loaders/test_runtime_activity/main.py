import json
import os
import resource

import requests


def get_peak_rss() -> int:
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


if __name__ == "__main__":
    errors: dict[str, str | None] = {}
    ram: dict[str, int] = {"start": get_peak_rss()}

    try:
        requests.get("https://www.google.com")
        ram["after_network"] = get_peak_rss()
    except Exception as e:
        errors["network"] = str(e)

    try:
        os.system("cat /etc/passwd > /dev/null")
        ram["after_cat"] = get_peak_rss()
    except Exception as e:
        errors["cat"] = str(e)

    print(
        json.dumps(
            {
                "ram": ram,
                "errors": errors,
            }
        )
    )
