from datetime import datetime
import os
import pathlib
import threading
import time

import docker as docker_og
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as
from rich import print

import dyana_cli.loaders as loaders
import dyana_cli.loaders.docker as docker
from dyana_cli.loaders.settings import LoaderSettings, ParsedArgument


class GpuDeviceUsage(BaseModel):
    device_index: int
    device_name: str
    total_memory: int
    free_memory: int


class Run(BaseModel):
    loader_name: str | None = None
    build_platform: str | None = None
    build_args: dict[str, str] | None = None
    arguments: list[str] | None = None
    volumes: dict[str, str] | None = None
    errors: dict[str, str] | None = None
    ram: dict[str, int] | None = None
    gpu: dict[str, list[GpuDeviceUsage]] | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None


class Loader:
    def __init__(self, name: str, timeout: int, platform: str | None, args: list[str] | None = None):
        # make sure that name does not include a path traversal
        if "/" in name or ".." in name:
            raise ValueError("Loader name cannot include a path traversal")

        self.image_name = name
        self.timeout = timeout
        self.path = os.path.join(loaders.__path__[0], name)
        self.reader_thread: threading.Thread | None = None
        self.container: docker.models.containers.Container | None = None
        self.container_id: str | None = None
        self.output: str = ""
        self.platform = platform
        self.settings_path = os.path.join(self.path, "settings.yml")
        self.build_args: dict[str, str] | None = None
        self.args: list[ParsedArgument] | None = None

        if os.path.exists(self.settings_path):
            with open(self.settings_path) as f:
                self.settings = parse_yaml_raw_as(LoaderSettings, f.read())
                self.build_args = self.settings.parse_build_args(args)
                self.args = self.settings.parse_args(args)
        else:
            self.settings = None

        self.dockerfile = os.path.join(self.path, "Dockerfile")
        if not os.path.exists(self.dockerfile):
            raise ValueError(f"Loader {name} does not exist")
        elif not os.path.isfile(self.dockerfile):
            raise ValueError(f"Loader {name} does not contain a Dockerfile")

        print(f":whale: [bold]loader[/]: initializing loader [bold]{name}[/]")

        self.name = name
        self.image_name = f"dyana-{name}-loader"
        self.image = docker.build(self.path, self.image_name, platform=self.platform, build_args=self.build_args)

        if self.platform:
            print(
                f":whale: [bold]loader[/]: using image [green]{self.image.tags[0]}[/] [dim]({self.image.id})[/] ({self.platform})"
            )
        else:
            print(f":whale: [bold]loader[/]: using image [green]{self.image.tags[0]}[/] [dim]({self.image.id})[/]")

    def _reader_thread(self):
        # attach to the container's logs with stream=True to get a generator
        logs = self.container.logs(stream=True, follow=True)

        # loop while the container is running
        while self.container.status in ["created", "running"]:
            # https://github.com/docker/docker-py/issues/2913
            for char in logs:
                try:
                    char = char.decode("utf-8")
                except UnicodeDecodeError:
                    char = char.decode("utf-8", errors="replace")

                self.output += char

            try:
                # refresh container status
                self.container.reload()
            except:
                # container is deleted
                break

    def run(self, allow_network: bool = False, allow_gpus: bool = True, allow_volume_write: bool = False) -> Run:
        volumes = {}
        arguments = []

        if self.args:
            for arg in self.args:
                arguments.append(f"--{arg.name}")

                # check if the argument is a volume
                if arg.volume:
                    volume_path = pathlib.Path(arg.value).resolve().absolute()
                    # NOTE: we need to preserve the folder name since AutoModel will use it to
                    # determine the model type, make it lowercase for matching
                    volume_name = volume_path.name.lower()
                    volume = f"/{volume_name}"
                    volumes[str(volume_path)] = volume

                    arguments.append(volume)
                else:
                    arguments.append(arg.value)

        if allow_network:
            print(
                ":popcorn: [bold]loader[/]: [yellow]warning: allowing bridged network access to the model container[/]"
            )

        if allow_volume_write:
            print(":popcorn: [bold]loader[/]: [yellow]warning: allowing volume write to the model container[/]")

        if arguments:
            print(f":popcorn: [bold]loader[/]: executing with arguments [dim]{arguments}[/] ...")
        else:
            print(":popcorn: [bold]loader[/]: executing ...")

        try:
            self.output = ""
            self.container = docker.run_detached(
                self.image, arguments, volumes, allow_network, allow_gpus, allow_volume_write
            )
            self.container_id = self.container.id
            self.reader_thread = threading.Thread(target=self._reader_thread)
            self.reader_thread.start()

            started_at = datetime.now()
            while self.container.status in ["created", "running"]:
                time.sleep(1.0)
                try:
                    # refresh container status
                    self.container.reload()
                except:
                    # container is deleted
                    break

                if (datetime.now() - started_at).total_seconds() > self.timeout:
                    self.container.kill()
                    print(":popcorn: [bold]loader[/]: [red]timeout reached, killing container[/]")
                    run = Run()
                    run.loader_name = self.name
                    run.build_platform = self.platform
                    run.build_args = self.build_args
                    run.arguments = arguments
                    run.volumes = volumes
                    run.errors = {"timeout": "timeout reached, killing container"}
                    return run

            if not self.output.startswith("{"):
                idx = self.output.find("{")
                if idx > 0:
                    before = self.output[:idx]
                    self.output = self.output[idx:]
                    print(f":popcorn: [bold]loader[/]: [dim]{before}[/]")

            try:
                run = Run.model_validate_json(self.output)
                run.loader_name = self.name
                run.build_platform = self.platform
                run.build_args = self.build_args
                run.arguments = arguments
                run.volumes = volumes
                return run
            except Exception as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: [bold red]{self.output}[/]")
                raise e

        except docker_og.errors.ContainerError as ce:
            print(f"\nContainer failed with exit code {ce.exit_status}")
            print("\nContainer output:")
            print(ce.stderr.decode("utf-8"))
