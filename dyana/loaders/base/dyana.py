import atexit
import os
import resource
import shutil
import sys
import typing as t
from contextlib import contextmanager
from io import StringIO


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


atexit.register(save_artifacts)


class Profiler:
    def __init__(self, gpu: bool = False):
        self._errors: dict[str, str] = {}
        self._disk: dict[str, int] = {"start": get_disk_usage()}
        self._ram: dict[str, int] = {"start": get_peak_rss()}
        self._gpu: dict[str, list[dict[str, t.Any]]] = {"start": get_gpu_usage()} if gpu else {}
        self._imports_at_start = get_current_imports()
        self._additionals: dict[str, t.Any] = {}

    def track_memory(self, event: str) -> None:
        self._ram[event] = get_peak_rss()
        if self._gpu:
            self._gpu[event] = get_gpu_usage()

    def track_disk(self, event: str) -> None:
        self._disk[event] = get_disk_usage()

    def track_error(self, event: str, error: str) -> None:
        self._errors[event] = error

    def track(self, key: str, value: t.Any) -> None:
        self._additionals[key] = value

    def as_dict(self) -> dict[str, t.Any]:
        imports_at_end = get_current_imports()
        imported = {k: imports_at_end[k] for k in imports_at_end if k not in self._imports_at_start}

        as_dict: dict[str, t.Any] = {
            "ram": self._ram,
            "disk": self._disk,
            "errors": self._errors,
            "extra": {"imports": imported},
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
    Get the peak RSS memory usage of the current process.
    """
    # https://stackoverflow.com/a/7669482
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


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
