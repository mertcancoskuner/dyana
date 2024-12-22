import os
import pathlib

import docker as docker_og
from pydantic import BaseModel
from rich import print

import dyana_cli.loaders as loaders
import dyana_cli.loaders.docker as docker


class RamUsage(BaseModel):
    start: int
    after_tokenizer_loaded: int | None = 0
    after_tokenization: int | None = 0
    after_model_loaded: int | None = 0
    after_model_inference: int | None = 0


class GpuDeviceUsage(BaseModel):
    device_index: int
    device_name: str
    total_memory: int
    free_memory: int


class GpuUsage(BaseModel):
    start: list[GpuDeviceUsage]
    after_tokenizer_loaded: list[GpuDeviceUsage] | None = []
    after_tokenization: list[GpuDeviceUsage] | None = []
    after_model_loaded: list[GpuDeviceUsage] | None = []
    after_model_inference: list[GpuDeviceUsage] | None = []


class Run(BaseModel):
    input: str
    errors: dict[str, str | None] | None = None
    ram: RamUsage
    gpu: GpuUsage


class Loader:
    def __init__(self, name: str):
        # make sure that name does not include a path traversal
        if "/" in name or ".." in name:
            raise ValueError("Loader name cannot include a path traversal")

        self.path = os.path.join(loaders.__path__[0], name)
        self.dockerfile = os.path.join(self.path, "Dockerfile")

        if not os.path.exists(self.dockerfile):
            raise ValueError(f"Loader {name} does not exist")
        elif not os.path.isfile(self.dockerfile):
            raise ValueError(f"Loader {name} does not contain a Dockerfile")

        print(f":whale: [bold]loader[/]: initializing loader [bold]{name}[/]")

        self.name = f"dyana-{name}-loader"
        self.image = docker.build(self.path, self.name)
        print(f":whale: [bold]loader[/]: using image [green]{self.image.tags[0]}[/] [dim]({self.image.id})[/]")

    def run(self, model_path: pathlib.Path, input: str, allow_network: bool = False, allow_gpus: bool = True) -> Run:
        model_path = model_path.resolve().absolute()
        # NOTE: we need to preserve the folder name since AutoModel will use it to
        # determine the model type, make it lowercase for matching
        model_name = model_path.name.lower()
        model_volume = f"/{model_name}"

        print(f":popcorn: [bold]loader[/]: executing inference for [bold]{model_path}[/] ...")

        try:
            out = docker.run(self.image, [model_volume, input], {model_path: model_volume}, allow_network, allow_gpus)
            if not out.startswith("{"):
                idx = out.find("{")
                if idx > 0:
                    before = out[:idx]
                    out = out[idx:]
                    print(f":popcorn: [bold]loader[/]: [dim]{before}[/]")

            try:
                return Run.model_validate_json(out)
            except Exception as e:
                print(f"Validation error: {e}")
                print(f"Invalid JSON: [bold red]{out}[/]")
                raise e

        except docker_og.errors.ContainerError as ce:
            print(f"\nContainer failed with exit code {ce.exit_status}")
            print("\nContainer output:")
            print(ce.stderr.decode("utf-8"))
