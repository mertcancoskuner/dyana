import argparse
import json
import shutil
import time
import typing
from typing import Any

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from dyana import Profiler  # type: ignore[attr-defined]


def collect_performance_metrics(driver: webdriver.Chrome) -> dict[str, Any]:
    """Collect detailed performance metrics using Chrome DevTools Protocol."""
    metrics = {}

    # Navigation Timing API metrics
    navigation_timing = driver.execute_script("""
        const performance = window.performance;
        const timing = performance.timing;
        return {
            'navigationStart': timing.navigationStart,
            'responseEnd': timing.responseEnd,
            'domComplete': timing.domComplete,
            'loadEventEnd': timing.loadEventEnd,
            'pageLoadTime': timing.loadEventEnd - timing.navigationStart,
            'dnsLookupTime': timing.domainLookupEnd - timing.domainLookupStart,
            'tcpConnectTime': timing.connectEnd - timing.connectStart,
            'serverResponseTime': timing.responseEnd - timing.requestStart,
            'domProcessingTime': timing.domComplete - timing.domLoading
        };
    """)
    metrics["timing"] = navigation_timing

    # Memory info
    memory_info = driver.execute_script("""
        return {
            'jsHeapSizeLimit': window.performance.memory.jsHeapSizeLimit,
            'totalJSHeapSize': window.performance.memory.totalJSHeapSize,
            'usedJSHeapSize': window.performance.memory.usedJSHeapSize
        };
    """)
    metrics["memory"] = memory_info

    # Resource timing data
    resource_timing = driver.execute_script("""
        return performance.getEntriesByType('resource').map(entry => ({
            name: entry.name,
            entryType: entry.entryType,
            startTime: entry.startTime,
            duration: entry.duration,
            initiatorType: entry.initiatorType
        }));
    """)
    metrics["resources"] = resource_timing

    return metrics


def analyze_page_content(driver: webdriver.Chrome) -> dict[str, Any]:
    """Analyze page content and structure."""
    return typing.cast(
        dict[str, Any],
        driver.execute_script("""
        return {
            'elements': document.getElementsByTagName('*').length,
            'images': document.getElementsByTagName('img').length,
            'links': document.getElementsByTagName('a').length,
            'scripts': document.getElementsByTagName('script').length,
            'styles': document.getElementsByTagName('link').length,
            'iframes': document.getElementsByTagName('iframe').length,
            'documentSize': document.documentElement.innerHTML.length,
        };
    """),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile website performance")
    parser.add_argument("--url", help="URL to open", required=True)
    parser.add_argument("--wait-for", help="CSS selector to wait for", default=None)
    parser.add_argument("--timeout", help="Timeout in seconds", type=int, default=30)
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
        # Enable performance logging
        chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})

        driver = webdriver.Chrome(options=chrome_options, service=Service(shutil.which("chromedriver")))
        driver.implicitly_wait(10)

        profiler.track_memory("before_load")

        start_time = time.time()
        driver.get(args.url)

        # Wait for specific element if requested
        if args.wait_for:
            try:
                WebDriverWait(driver, args.timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, args.wait_for))
                )
            except TimeoutException:
                profiler.track_error("wait", f"Timeout waiting for element: {args.wait_for}")

        load_time = time.time() - start_time
        profiler.track_memory("after_load")

        # Collect performance metrics
        try:
            metrics = collect_performance_metrics(driver)
            content_analysis = analyze_page_content(driver)

            # Get console logs
            console_logs = driver.get_log("browser")

            # Get network logs
            network_logs = driver.get_log("performance")

            # Add all metrics to profiler
            profiler.extra = {
                "load_time": load_time,
                "performance_metrics": metrics,
                "content_analysis": content_analysis,
                "console_logs": console_logs,
                "network_logs": network_logs,
                "title": driver.title,
                "url": driver.current_url,
                "status_code": driver.execute_script("return window.performance.getEntries()[0].responseStatus"),
            }
        except Exception as e:
            profiler.track_error("metrics", str(e))

        # Take screenshot
        try:
            screenshot = driver.get_screenshot_as_base64()
            profiler.extra["screenshot"] = screenshot
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
