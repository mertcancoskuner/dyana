import os
import pathlib
import shutil
import threading
import time
import typing as t
from datetime import datetime

import docker as docker_pkg
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as
from rich import print

import dyana.docker as docker
import dyana.loaders as loaders
from dyana.loaders.settings import LoaderSettings, ParsedArgument


class GpuDeviceUsage(BaseModel):
    device_index: int
    device_name: str
    total_memory: int
    free_memory: int


class NetworkDeviceUsage(BaseModel):
    rx: int
    tx: int


class Run(BaseModel):
    loader_name: str | None = None
    build_platform: str | None = None
    build_args: dict[str, str] | None = None
    arguments: list[str] | None = None
    volumes: dict[str, str] | None = None
    errors: dict[str, str] | None = None
    ram: dict[str, int] | None = None
    gpu: dict[str, list[GpuDeviceUsage]] | None = None
    disk: dict[str, int] | None = None
    network: dict[str, dict[str, NetworkDeviceUsage]] | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    extra: dict[str, t.Any] | None = None


class Loader:
    def __init__(
        self,
        name: str,
        timeout: int = 600,
        build: bool = True,
        platform: str | None = None,
        args: list[str] | None = None,
        save: list[str] | None = None,
        save_to: pathlib.Path = pathlib.Path("./artifacts"),
        verbose: bool = False,
    ):
        # make sure that name does not include a path traversal
        if "/" in name or ".." in name:
            raise ValueError("Loader name cannot include a path traversal")

        self.image_name = name
        self.timeout = timeout
        self.base_lib_path = os.path.join(loaders.__path__[0], "base/dyana.py")
        self.path = os.path.join(loaders.__path__[0], name)
        self.reader_thread: threading.Thread | None = None
        self.container: docker_pkg.models.containers.Container | None = None
        self.container_id: str | None = None
        self.output: str = ""
        self.platform = platform
        self.settings_path = os.path.join(self.path, "settings.yml")
        self.settings: LoaderSettings | None = None
        self.build_args: dict[str, str] | None = None
        self.args: list[ParsedArgument] | None = None
        self.save: list[str] | None = save
        self.save_to: pathlib.Path = save_to.resolve().absolute()

        if os.path.exists(self.settings_path):
            with open(self.settings_path) as f:
                self.settings = parse_yaml_raw_as(LoaderSettings, f.read())
                if args:
                    self.build_args = self.settings.parse_build_args(args)
                    self.args = self.settings.parse_args(args)
        else:
            self.settings = None

        self.dockerfile = os.path.join(self.path, "Dockerfile")
        if not os.path.exists(self.dockerfile):
            raise ValueError(f"Loader {name} does not exist, use [bold]dyana loaders[/] to see the available loaders")
        elif not os.path.isfile(self.dockerfile):
            raise ValueError(f"Loader {name} does not contain a Dockerfile")

        self.name = name
        self.image_name = f"dyana-{name}-loader"

        if build:
            if self.settings and self.settings.args:
                for arg in self.settings.args:
                    if arg.required and not self.args:
                        raise ValueError(f"Argument --{arg.name} is required")

            print(f":whale: [bold]loader[/]: initializing loader [bold]{name}[/]")
            # copy the base dyana.py to the loader directory
            # TODO: ideally the file name should be randomized in order to avoid low-hanging fruits in terms
            # of sandbox detection techniques
            shutil.copy(self.base_lib_path, os.path.join(self.path, "dyana.py"))

            self.image = docker.build(
                self.path, self.image_name, platform=self.platform, build_args=self.build_args, verbose=verbose
            )
            if self.platform:
                print(
                    f":whale: [bold]loader[/]: using image [green]{self.image.tags[0]}[/] [dim]({self.image.id})[/] ({self.platform})"
                )
            # else:
            # print(f":whale: [bold]loader[/]: using image [green]{self.image.tags[0]}[/] [dim]({self.image.id})[/]")

    def _reader_thread(self) -> None:
        if not self.container:
            raise Exception("Container not created")

        # attach to the container's logs with stream=True to get a generator
        logs = self.container.logs(stream=True, follow=True, stdout=True, stderr=True)

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
            except Exception:
                # container is deleted
                break

    def _create_errored_run(self, error_key: str, error_message: str) -> Run:
        run = Run()
        run.loader_name = self.name
        run.build_platform = self.platform
        run.build_args = self.build_args
        run.arguments = [arg.value for arg in self.args] if self.args else None
        run.errors = {error_key: error_message}
        return run

    def run(self, allow_network: bool = False, allow_gpus: bool = True, allow_volume_write: bool = False) -> Run:
        volumes = {}
        arguments = []

        if self.args:
            for arg in self.args:
                the_arg = f"--{arg.name}"
                arguments.append(the_arg)

                # check if the argument is a volume
                if arg.volume:
                    volume_path = pathlib.Path(arg.value).resolve().absolute()
                    # NOTE: we need to preserve the folder name since AutoModel will use it to
                    # determine the model type, make it lowercase for matching
                    volume_name = volume_path.name.lower()
                    volume = f"/{volume_name}"
                    volumes[str(volume_path)] = volume

                    arguments.append(volume)
                elif arg.value != the_arg:
                    arguments.append(arg.value)

        if self.settings and self.settings.network:
            allow_network = True
            print(":popcorn: [bold]loader[/]: [yellow]required bridged network access[/]")

        elif allow_network:
            print(":popcorn: [bold]loader[/]: [yellow]warning: allowing bridged network access to the container[/]")

        if allow_volume_write:
            print(":popcorn: [bold]loader[/]: [yellow]warning: allowing volume write to the container[/]")

        if arguments:
            print(f":popcorn: [bold]loader[/]: executing with arguments [dim]{arguments}[/] ...")
        else:
            print(":popcorn: [bold]loader[/]: executing ...")

        try:
            self.output = ""
            environment = {}
            if self.save:
                environment["DYANA_SAVE"] = ",".join(self.save)
                volumes[str(self.save_to)] = "/artifacts"
                if not os.path.exists(self.save_to):
                    os.makedirs(self.save_to)

                print(f":popcorn: [bold]loader[/]: saving artifacts to [dim]{self.save_to}[/]")

            self.container = docker.run_detached(
                self.image,
                arguments,
                volumes,
                environment=environment,
                allow_network=allow_network,
                allow_gpus=allow_gpus,
                allow_volume_write=allow_volume_write,
            )
            self.container_id = self.container.id
            self.reader_thread = threading.Thread(target=self._reader_thread)
            self.reader_thread.start()

            started_at = datetime.now()
            while self.container.status in ["created", "running", "removing"]:
                time.sleep(1.0)
                try:
                    # refresh container status
                    self.container.reload()
                except Exception:
                    # container is deleted
                    break

                if (datetime.now() - started_at).total_seconds() > self.timeout:
                    self.container.kill()
                    print(":popcorn: [bold]loader[/]: [red]timeout reached, killing container[/]")
                    return self._create_errored_run("timeout", "timeout reached, killing container")

            extra_output: str | None = None
            if not self.output.startswith("{"):
                idx = self.output.find("{")
                if idx > 0:
                    extra_output = self.output[:idx]
                    self.output = self.output[idx:]
                    # print(f":popcorn: [bold]loader[/]: [dim]{extra_output}[/]")

            try:
                run = Run.model_validate_json(self.output)
                run.loader_name = self.name
                run.build_platform = self.platform
                run.build_args = self.build_args
                run.arguments = arguments
                run.volumes = volumes

                if extra_output:
                    if run.stdout:
                        run.stdout += extra_output
                    else:
                        run.stdout = extra_output
                return run
            except Exception as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: [bold red]{self.output}[/]")
                raise e

        except docker_pkg.errors.ContainerError as ce:
            print(f"\nContainer failed with exit code {ce.exit_status}")
            print("\nContainer output:")
            print(ce.stderr.decode("utf-8"))
            return self._create_errored_run("container_execution_error", ce.stderr.decode("utf-8"))
