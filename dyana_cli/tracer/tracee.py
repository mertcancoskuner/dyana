import json
import platform
import threading
import time
from datetime import datetime

from pydantic import BaseModel
from rich import print

import dyana_cli.loaders.docker as docker
from dyana_cli.loaders.loader import Loader, Run


class Trace(BaseModel):
    started_at: datetime
    ended_at: datetime
    platform: str
    run: Run
    events: list[dict] = []


class Tracer:
    DOCKER_IMAGE = "aquasec/tracee:latest"

    SECURITY_EVENTS: list[str] = [
        # cd tracee/signatures/go && grep -r "EventName:" --exclude="*_test.go" * | cut -d'"' -f2 | sort -u
        "anti_debugging",
        "aslr_inspection",
        "cgroup_notify_on_release",
        "cgroup_release_agent",
        "core_pattern_modification",
        "default_loader_mod",
        "disk_mount",
        "docker_abuse",
        "dropped_executable",
        "dynamic_code_loading",
        "fileless_execution",
        "hidden_file_created",
        "illegitimate_shell",
        "k8s_api_connection",
        "k8s_cert_theft",
        # Error: invalid event to trace: k8s_service_account_token
        # "k8s_service_account_token",
        "kernel_module_loading",
        "ld_preload",
        "proc_fops_hooking",
        "proc_kcore_read",
        "proc_mem_access",
        "proc_mem_code_injection",
        "process_vm_write_inject",
        "ptrace_code_injection",
        "rcd_modification",
        "sched_debug_recon",
        "scheduled_task_mod",
        "stdio_over_socket",
        "sudoers_modification",
        "syscall_hooking",
        "system_request_key_mod",
        # non signature related but still security related
        "hidden_kernel_module",
        "bpf_attach",
        "ftrace_hook",
        "hooked_syscall",
    ]

    DEFAULT_EVENTS: list[str] = [
        "security_file_open",
        "sched_process_exec",
        "security_socket_*",
        "net_packet_dns",
    ] + SECURITY_EVENTS

    def __init__(self, loader: Loader):
        print(":eye_in_speech_bubble:  [bold]tracer[/]: initializing ...")

        docker.pull(Tracer.DOCKER_IMAGE)

        self.loader = loader
        self.errors: list[str] = []
        self.trace: list[dict] = []
        self.args = [
            "--output",
            "json",
            # only trace events that are part of a new container
            "--scope",
            "container=new",
            # enable debug logging to know when tracee is ready
            "--log",
            "debug",
        ]
        for event in Tracer.DEFAULT_EVENTS:
            self.args.append("--events")
            self.args.append(event)

        self.reader_thread: threading.Thread | None = None
        self.container: docker.models.containers.Container | None = None
        self.ready = False

    def _reader_thread(self):
        # attach to the container's logs with stream=True to get a generator
        logs = self.container.logs(stream=True, follow=True)
        line = ""

        # loop while the container is running
        while self.container.status in ["created", "running"]:
            # https://github.com/docker/docker-py/issues/2913
            for char in logs:
                try:
                    char = char.decode("utf-8")
                except UnicodeDecodeError:
                    char = char.decode("utf-8", errors="replace")

                line += char
                if char == "\n":
                    self._on_tracer_event(line)
                    line = ""
            try:
                # refresh container status
                self.container.reload()
            except:
                # container is deleted
                break

    def _on_tracer_event(self, line: str) -> None:
        line = line.strip()
        if not line:
            return

        if not line.startswith("{"):
            print(f"[dim]{line}[/]")
            return

        message = json.loads(line)

        if "L" in message:
            if message["L"] == "DEBUG":
                # these are debug messages, do not collect them
                if "is ready callback" in line:
                    self.ready = True
            else:
                # other messages
                print(f":eye_in_speech_bubble:  [bold]tracer[/]: {message['M'].strip()}")

        elif "level" in message:
            # other messages
            if message["level"] in ["fatal", "error"]:
                err = message["error"].strip()
                print(f":exclamation: [bold red]tracer error:[/]: {err}")
                self.errors.append(err)
            else:
                msg = message["msg"].strip()
                print(f":eye_in_speech_bubble:  [bold]tracer[/]: {msg}")
        else:
            # actual events
            self.trace.append(message)

    def _start(self) -> None:
        self.errors.clear()
        self.trace.clear()

        # TODO: investigate these errors:
        # 👁️‍🗨️  tracer: KConfig: could not check enabled kconfig features
        # 👁️‍🗨️  tracer: KConfig: assuming kconfig values, might have unexpected behavior

        # start tracee in a detached container
        self.container = docker.run_privileged_detached(
            Tracer.DOCKER_IMAGE,
            self.args,
            volumes={"/etc/os-release": "/etc/os-release-host", "/var/run/docker.sock": "/var/run/docker.sock"},
            # override the entrypoint so we can pass our own arguments
            entrypoint="/tracee/tracee",
            environment={"LIBBPFGO_OSRELEASE_FILE": "/etc/os-release-host"},
        )

        # start reading tracee output in a separate thread
        self.reader_thread = threading.Thread(target=self._reader_thread)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        # tracee takes a few seconds to warm up and trace events
        while not self.ready:
            time.sleep(1)

    def _stop(self) -> None:
        print(":eye_in_speech_bubble:  [bold]tracer[/]: stopping ...")
        self.container.stop()

    def run_trace(
        self, allow_network: bool = False, allow_gpus: bool = True, allow_volume_write: bool = False
    ) -> Trace:
        self._start()

        print(":eye_in_speech_bubble:  [bold]tracer[/]: started ...")

        started_at = datetime.now()
        run = self.loader.run(allow_network, allow_gpus, allow_volume_write)
        ended_at = datetime.now()

        self._stop()

        if self.errors:
            run.errors["tracer"] = self.errors

        return Trace(
            platform=platform.platform(),
            started_at=started_at,
            ended_at=ended_at,
            # filter out any events from containers different than the one we created
            events=[event for event in self.trace if event["containerId"].lower() == self.loader.container_id.lower()],
            run=run,
        )
