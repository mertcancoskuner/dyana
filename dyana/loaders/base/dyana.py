import resource
import sys
import typing as t
from contextlib import contextmanager
from io import StringIO


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
