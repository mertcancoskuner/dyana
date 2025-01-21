import argparse
import json
import os
import shutil

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from dyana import Profiler  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile website performance")
    parser.add_argument("--url", help="URL to open", required=True)
    parser.add_argument("--wait-for", help="CSS selector to wait for", default=None)
    parser.add_argument(
        "--wait-for-timeout", help="Timeout to wait for the CSS selectorin seconds", type=int, default=30
    )
    parser.add_argument("--screenshot", help="Save a screenshot of the page", action="store_true")
    args = parser.parse_args()

    # Normalize URL by adding https:// if protocol is missing
    if "://" not in args.url:
        args.url = f"https://{args.url}"

    profiler: Profiler = Profiler()

    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        service = Service(executable_path=shutil.which("chromedriver"))
        service.start()

        driver = webdriver.Chrome(options=chrome_options, service=service)
        driver.implicitly_wait(10)

        profiler.track_memory("before_load")

        driver.get(args.url)

        if args.wait_for:
            # Wait for specific element if requested
            try:
                WebDriverWait(driver, args.wait_for_timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, args.wait_for))
                )
            except TimeoutException:
                profiler.track_error("wait", f"Timeout waiting for element: {args.wait_for}")

        profiler.track_memory("after_load")

        if args.screenshot:
            try:
                driver.get_screenshot_as_file("/tmp/screenshot.png")
                os.environ["DYANA_SAVE"] = "/tmp/screenshot.png"
            except Exception as e:
                profiler.track_error("screenshot", str(e))

        profiler.track_memory("after_profiling")

    except Exception as e:
        profiler.track_error("chrome", str(e))
    finally:
        try:
            driver.quit()
            profiler.track_memory("after_quit")
        except Exception:
            pass

    print(json.dumps(profiler.as_dict()))
