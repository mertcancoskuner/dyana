import json
import pathlib
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable

from pydantic import BaseModel
from rich import print

import dyana_cli.loaders.docker as docker
from dyana_cli.loaders.loader import GpuUsage, Loader, RamUsage


class AsyncProcessRunner:
    def __init__(self, command: list[str], on_output: Callable[[str], None]):
        self.command = command
        self.on_output = on_output
        self.process: subprocess.Popen | None = None
        self.output_thread: threading.Thread | None = None
        self.is_running = False

    def _read_output(self):
        while self.is_running:
            if self.process and self.process.stdout:
                line = self.process.stdout.readline()
                if line:
                    self.on_output(line.decode().rstrip())
                else:
                    break

    def start(self):
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=False
        )
        self.is_running = True
        self.output_thread = threading.Thread(target=self._read_output)
        self.output_thread.daemon = True
        self.output_thread.start()

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.output_thread:
            self.output_thread.join()

    def is_alive(self) -> bool:
        return self.process.poll() is None if self.process else False


class Trace(BaseModel):
    started_at: datetime
    ended_at: datetime
    model_path: str
    model_input: str
    errors: dict[str, list[str]] | None = None
    events: list[dict] = []
    ram: RamUsage
    gpu: GpuUsage


class Tracer:
    TRACEE_IMAGE = "aquasec/tracee:latest"

    # TODO: do we want to trace other events? https://aquasecurity.github.io/tracee/latest/docs/flags/events.1/
    DEFAULT_EVENTS: list[str] = ["security_file_open", "sched_process_exec", "security_socket_*"]

    def __init__(self, loader: Loader, events: list[str] = DEFAULT_EVENTS):
        self.docker = shutil.which("docker")
        if not self.docker:
            raise Exception("docker not found")

        print(":eye_in_speech_bubble:  [bold]tracer[/]: initializing ...")

        docker.pull(Tracer.TRACEE_IMAGE)

        self.loader = loader
        self.events = events
        self.errors: list[str] = []
        self.trace: list[dict] = []
        # TODO: change this to a detached docker.run
        self.args = [
            self.docker,
            "run",
            "--rm",
            "--pid=host",
            "--cgroupns=host",
            "--privileged",
            "-v",
            "/etc/os-release:/etc/os-release-host:ro",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-e",
            "LIBBPFGO_OSRELEASE_FILE=/etc/os-release-host",
            # override the entrypoint so we can pass our own arguments
            "--entrypoint",
            "/tracee/tracee",
            Tracer.TRACEE_IMAGE,
            "--output",
            "json",
            # only trace events that are part of a new container
            # TODO: find a more specific way to trace from the new container
            "--scope",
            "container=new",
        ]
        for event in events:
            self.args.append("--events")
            self.args.append(event)

        self.runner = AsyncProcessRunner(self.args, self._on_output)

    def _on_output(self, line: str) -> None:
        line = line.strip()
        if not line:
            return

        if not line.startswith("{"):
            print(f"[dim]{line}[/]")
            return

        # TODO: investigate possible KConfig fix for messages:
        #  KConfig: could not check enabled kconfig features
        #  KConfig: assuming kconfig values, might have unexpected behavior

        message = json.loads(line)
        if "level" in message:
            if message["level"] in ["fatal", "error"]:
                err = message["error"].strip()
                print(f":exclamation: [bold red]tracer error:[/]: {err}")
                self.errors.append(err)
            else:
                msg = message["msg"].strip()
                print(f":eye_in_speech_bubble:  [bold]tracer[/]: {msg}")
        else:
            self.trace.append(message)

    def _start(self) -> None:
        self.errors.clear()
        self.trace.clear()

        self.runner.start()

        # TODO: tracee takes a few seconds to warm up and trace events, is there a better way to wait for it?
        print(":eye_in_speech_bubble:  [bold]tracer[/]: warming up ...")
        for _ in range(10, 0, -1):
            time.sleep(1)

    def _stop(self) -> None:
        print(":eye_in_speech_bubble:  [bold]tracer[/]: stopping ...")
        self.runner.stop()

    def run_trace(
        self, model_path: pathlib.Path, model_input: str, allow_network: bool = False, allow_gpus: bool = True
    ) -> Trace:
        self._start()

        events = [f"[yellow]{e}[/]" for e in self.events]
        print(f":eye_in_speech_bubble:  [bold]tracer[/]: tracing {', '.join(events)} ...")

        started_at = datetime.now()
        run = self.loader.run(model_path, model_input, allow_network, allow_gpus)
        ended_at = datetime.now()

        self._stop()

        # TODO: filter out any events from containers different than the one we created

        # consolidate in a single Trace object
        errors: dict[str, str] = {}
        for error, message in run.errors.items():
            if message:
                errors[error] = [message]

        if self.errors:
            errors["tracer"] = self.errors

        return Trace(
            model_path=str(model_path.resolve().absolute()),
            started_at=started_at,
            ended_at=ended_at,
            model_input=model_input,
            errors=errors if errors else None,
            events=self.trace,
            ram=run.ram,
            gpu=run.gpu,
        )
