import atexit
import json
import os
import shutil
import sys
import typing as t
from contextlib import contextmanager
from io import StringIO

import psutil


def save_artifacts() -> None:
    artifacts = os.environ.get("DYANA_SAVE", "").split(",")
    if artifacts:
        for artifact in artifacts:
            try:
                if os.path.isdir(artifact):
                    shutil.copytree(artifact, f"/artifacts/{artifact}")
                elif os.path.isfile(artifact):
                    shutil.copy(artifact, "/artifacts")
            except Exception:
                pass


class Profiler:
    instance: t.Optional["Profiler"] = None

    @staticmethod
    def flush() -> None:
        if Profiler.instance:
            # add a prefix to the output to make it easier to identify in the logs
            print("<DYANA_PROFILE>" + json.dumps(Profiler.instance.as_dict()))

    def __init__(self, gpu: bool = False):
        self._errors: dict[str, str] = {}
        self._warnings: dict[str, str] = {}
        self._disk: dict[str, int] = {"start": get_disk_usage()}
        self._ram: dict[str, int] = {"start": get_peak_rss()}
        self._gpu: dict[str, list[dict[str, t.Any]]] = {"start": get_gpu_usage()} if gpu else {}
        self._network: dict[str, dict[str, dict[str, int]]] = {"start": get_network_stats()}
        self._imports_at_start = get_current_imports()
        self._additionals: dict[str, t.Any] = {}
        self._extra: dict[str, t.Any] = {}

        Profiler.instance = self

    def track_memory(self, event: str) -> None:
        self._ram[event] = get_peak_rss()
        if self._gpu:
            self._gpu[event] = get_gpu_usage()

    def track_disk(self, event: str) -> None:
        self._disk[event] = get_disk_usage()

    def track_network(self, event: str) -> None:
        self._network[event] = get_network_stats()

    def track_error(self, event: str, error: str) -> None:
        self._errors[event] = error

    def track_warning(self, event: str, warning: str) -> None:
        self._warnings[event] = warning

    def track(self, key: str, value: t.Any) -> None:
        self._additionals[key] = value

    def track_extra(self, key: str, value: t.Any) -> None:
        self._extra[key] = value

    def as_dict(self) -> dict[str, t.Any]:
        imports_at_end = get_current_imports()
        imported = {k: imports_at_end[k] for k in imports_at_end if k not in self._imports_at_start}

        if len(self._network.keys()) == 1:
            self.track_network("end")

        as_dict: dict[str, t.Any] = {
            "ram": self._ram,
            "disk": self._disk,
            "network": self._network,
            "errors": self._errors,
            "warnings": self._warnings,
            "extra": {"imports": imported, **self._extra},
        } | self._additionals

        if self._gpu:
            as_dict["gpu"] = self._gpu

        return as_dict


@contextmanager
def capture_output() -> t.Generator[tuple[StringIO, StringIO], None, None]:
    """
    Context manager to capture stdout and stderr

    Returns:
        tuple: (stdout_content, stderr_content)
    """
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_disk_usage() -> int:
    """
    Get the disk usage.
    """
    _, used, _ = shutil.disk_usage("/")
    return used


def get_peak_rss() -> int:
    """
    Get the combined RSS memory usage of the current process and all its child processes.
    """
    loader_process: psutil.Process = psutil.Process()
    loader_rss: int = loader_process.memory_info().rss
    children_rss: int = 0

    for child in loader_process.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue

    return loader_rss + children_rss


def get_gpu_usage() -> list[dict[str, t.Any]]:
    """
    Get the GPU usage, for each GPU, of the current process.
    """
    import torch

    usage: list[dict[str, t.Any]] = []

    if torch.cuda.is_available():
        # for each GPU
        for i in range(torch.cuda.device_count()):
            dev = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            (free, total) = mem

            usage.append(
                {
                    "device_index": i,
                    "device_name": dev.name,
                    "total_memory": total,
                    "free_memory": free,
                }
            )

    return usage


def get_current_imports() -> dict[str, str | None]:
    """
    Get the currently imported modules.
    """
    imports: dict[str, str | None] = {}

    # for each loaded module
    for module_name, module in sys.modules.items():
        if module:
            imports[module_name] = module.__dict__["__file__"] if "__file__" in module.__dict__ else None

    return imports


def get_network_stats() -> dict[str, dict[str, int]]:
    """
    Parse /proc/net/dev and return a dictionary of network interface statistics.
    Returns a dictionary where each key is an interface name and each value is
    a dictionary containing bytes_received and bytes_sent.
    """
    stats: dict[str, dict[str, int]] = {}

    with open("/proc/net/dev") as f:
        # skip the first two header lines
        next(f)
        next(f)

        for line in f:
            # split the line into interface name and statistics
            parts = line.strip().split(":")
            if len(parts) != 2:
                continue

            interface = parts[0].strip()
            values = parts[1].split()
            stats[interface] = {"rx": int(values[0]), "tx": int(values[8])}

    return stats


# register atexit handlers
atexit.register(save_artifacts)
atexit.register(Profiler.flush)
