import json
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class TraceResult:
    url: str
    trace_file: pathlib.Path
    start_time: datetime
    end_time: datetime | None = None
    timeout: bool = False
    dns_events: int = 0
    total_events: int = 0
    duration: float = 0.0


def run_trace(url: str, output_dir: pathlib.Path, output_log: pathlib.Path) -> TraceResult:
    # This is where the JSON trace data will be saved
    trace_file = output_dir / f"trace_{url.replace('://', '_').replace('/', '_')}.json"
    result = TraceResult(url=url, trace_file=trace_file, start_time=datetime.now())

    try:
        # Run trace and capture its output
        process = subprocess.run(
            ["dyana", "trace", "--loader", "website", "--url", url, "--output", str(trace_file)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Append the raw stdout to our consolidated log file
        with open(output_log, "a") as f:
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"Trace for {url} started at {result.start_time}\n")
            f.write(f"{'=' * 80}\n")
            f.write(process.stdout)
            if process.stderr:
                f.write(f"\nSTDERR:\n{process.stderr}")

        # Process the trace data for stats
        with open(trace_file) as f:
            trace_data = json.load(f)

        # Count DNS events
        dns_events = 0
        for event in trace_data.get("events", []):
            if "dns" in str(event).lower():
                dns_events += 1

        result.dns_events = dns_events
        result.total_events = len(trace_data.get("events", []))
        result.timeout = any(
            error.lower().startswith("timeout") for error in trace_data.get("run", {}).get("errors", {}).values()
        )

    except subprocess.CalledProcessError as e:
        with open(output_log, "a") as f:
            f.write(f"\nError running trace for {url}:\n")
            f.write(f"stdout: {e.stdout}\n")
            f.write(f"stderr: {e.stderr}\n")
        result.timeout = True

    result.end_time = datetime.now()
    result.duration = (result.end_time - result.start_time).total_seconds()
    return result


def run_concurrent_tests(urls: List[str], concurrent_runs: int = 3, iterations: int = 2) -> List[TraceResult]:
    output_dir = pathlib.Path("./trace_results")
    output_dir.mkdir(exist_ok=True)
    output_log = output_dir / "raw_output.log"

    # Initialize the consolidated log file
    with open(output_log, "w") as f:
        f.write(f"Website Loader Test Run\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Testing URLs: {', '.join(urls)}\n")
        f.write(f"Iterations: {iterations}, Concurrent runs: {concurrent_runs}\n")

    results = []

    print(f"\nRunning {len(urls)} URLs x {iterations} iterations with {concurrent_runs} concurrent traces")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=concurrent_runs) as executor:
        futures = []
        for _ in range(iterations):
            for url in urls:
                futures.append(executor.submit(run_trace, url, output_dir, output_log))

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"\nTrace completed for {result.url}:")
            print(f"  Duration    : {result.duration:.2f}s")
            print(f"  DNS Events  : {result.dns_events}")
            print(f"  Total Events: {result.total_events}")
            print(f"  Timeout     : {result.timeout}")

    return results


def analyze_results(results: List[TraceResult]) -> None:
    print("\nAnalysis")
    print("=" * 80)

    # Group results by URL
    url_results = {}
    for result in results:
        if result.url not in url_results:
            url_results[result.url] = []
        url_results[result.url].append(result)

    for url, traces in url_results.items():
        timeouts = sum(1 for t in traces if t.timeout)
        traces_with_dns = sum(1 for t in traces if t.dns_events > 0)
        avg_dns_events = sum(t.dns_events for t in traces) / len(traces)
        avg_duration = sum(t.duration for t in traces) / len(traces)

        print(f"\nURL: {url}")
        print(f"  Total Traces : {len(traces)}")
        print(f"  Timeouts     : {timeouts}")
        print(f"  DNS Observed : {traces_with_dns}/{len(traces)} traces")
        print(f"  Avg DNS Events: {avg_dns_events:.1f}")
        print(f"  Avg Duration : {avg_duration:.2f}s")


if __name__ == "__main__":
    # Test URLs - mix of different protocols and formats
    urls = ["facebook.com", "https://facebook.com", "http://facebook.com", "google.com", "github.com"]

    results = run_concurrent_tests(urls, concurrent_runs=3, iterations=2)
    analyze_results(results)
    print(f"\nRaw trace output available at: ./trace_results/raw_output.log")
