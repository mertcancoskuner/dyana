import atexit
import json
import os
import shutil
import sys
import time
import typing as t
from contextlib import contextmanager
from io import StringIO

from pydantic import BaseModel


class GpuDeviceUsage(BaseModel):
    device_index: int
    device_name: str
    total_memory: int
    free_memory: int


class NetworkDeviceUsage(BaseModel):
    rx: int
    tx: int


class Stage(BaseModel):
    # stage name
    name: str
    # stage timestamp
    timestamp: int
    # current memory usage
    ram: int
    # GPU memory usage if GPU is available
    gpu: list[GpuDeviceUsage] | None = None
    # disk usage
    disk: int
    # network usage for each interface
    network: dict[str, NetworkDeviceUsage]
    # newly imported modules
    imports: dict[str, str | None]

    @staticmethod
    def create(name: str, prev_imports: dict[str, str | None] | None = None, with_gpu: bool = False) -> "Stage":
        timestamp = time.time_ns()
        ram = get_peak_rss()
        gpu = get_gpu_usage() if with_gpu else None
        disk = get_disk_usage()
        network = get_network_stats()
        current_imports = get_current_imports()
        if prev_imports is None:
            imports = current_imports
        else:
            imports = {k: current_imports[k] for k in current_imports if k not in prev_imports}

        return Stage(
            name=name,
            timestamp=timestamp,
            ram=ram,
            gpu=gpu,
            disk=disk,
            network=network,
            imports=imports,
        )


class Profiler:
    instance: t.Optional["Profiler"] = None

    @staticmethod
    def flush() -> None:
        if Profiler.instance:
            # add a prefix to the output to make it easier to identify in the logs
            print("<DYANA_PROFILE>" + json.dumps(Profiler.instance.as_dict()))

    def __init__(self, gpu: bool = False):
        self._gpu = gpu
        self._errors: dict[str, str] = {}
        self._warnings: dict[str, str] = {}
        self._stages: list[Stage] = [Stage.create("start", with_gpu=gpu)]
        self._additionals: dict[str, t.Any] = {}
        self._extra: dict[str, t.Any] = {}

        Profiler.instance = self

    def on_stage(self, name: str) -> None:
        # collect all imports from previous stages in order to only track newly imported modules
        prev_imports = {}
        for stage in self._stages:
            for k, v in stage.imports.items():
                prev_imports[k] = v

        self._stages.append(Stage.create(name, prev_imports=prev_imports, with_gpu=self._gpu))

    def track_error(self, event: str, error: str) -> None:
        self._errors[event] = error

    def track_warning(self, event: str, warning: str) -> None:
        self._warnings[event] = warning

    def track(self, key: str, value: t.Any) -> None:
        self._additionals[key] = value

    def track_extra(self, key: str, value: t.Any) -> None:
        self._extra[key] = value

    def as_dict(self) -> dict[str, t.Any]:
        self.on_stage("end")

        as_dict: dict[str, t.Any] = {
            "stages": [stage.model_dump() for stage in self._stages],
            "errors": self._errors,
            "warnings": self._warnings,
            "extra": self._extra,
        } | self._additionals

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
    import psutil

    loader_process: psutil.Process = psutil.Process()
    loader_rss: int = loader_process.memory_info().rss
    children_rss: int = 0

    for child in loader_process.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue

    return loader_rss + children_rss


def get_gpu_usage() -> list[GpuDeviceUsage]:
    """
    Get the GPU usage, for each GPU, of the current process.
    """
    import torch

    usage: list[GpuDeviceUsage] = []

    if torch.cuda.is_available():
        # for each GPU
        for i in range(torch.cuda.device_count()):
            dev = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            (free, total) = mem

            usage.append(
                GpuDeviceUsage(
                    device_index=i,
                    device_name=dev.name,
                    total_memory=total,
                    free_memory=free,
                )
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


def get_network_stats() -> dict[str, NetworkDeviceUsage]:
    """
    Parse /proc/net/dev and return a dictionary of network interface statistics.
    Returns a dictionary where each key is an interface name and each value is
    a dictionary containing bytes_received and bytes_sent.
    """
    stats: dict[str, NetworkDeviceUsage] = {}

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
            stats[interface] = NetworkDeviceUsage(rx=int(values[0]), tx=int(values[8]))

    return stats


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


# register atexit handlers
atexit.register(save_artifacts)
atexit.register(Profiler.flush)
