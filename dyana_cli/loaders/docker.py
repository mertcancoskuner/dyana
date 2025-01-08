import pathlib
import re

import docker
from docker.models.images import Image
from rich import print

try:
    client = docker.from_env()
except docker.errors.DockerException:
    client = None


def sanitized_agent_name(name: str) -> str:
    """
    Sanitizes an agent name to be used as a Docker repository name.
    """

    # convert to lowercase
    name = name.lower()
    # replace non-alphanumeric characters with hyphens
    name = re.sub(r"[^\w\s-]", "", name)
    # replace one or more whitespace characters with a single hyphen
    name = re.sub(r"[-\s]+", "-", name)
    # remove leading or trailing hyphens
    name = name.strip("-")

    return name


def build(
    directory: str | pathlib.Path,
    name: str,
    platform: str | None = None,
    build_args: dict[str, str] | None = None,
    force_rebuild: bool = False,
) -> Image:
    if client is None:
        raise Exception("Docker not available")

    norm_name = sanitized_agent_name(name)
    if norm_name != name:
        print(f"[yellow]Warning:[/] sanitized agent name from '{name}' to '{norm_name}'")
        name = norm_name

    id: str | None = None
    for item in client.api.build(
        path=str(directory),
        tag=name,
        decode=True,
        nocache=force_rebuild,
        pull=force_rebuild,
        buildargs=build_args,
        platform=platform,
        # remove intermediate containers
        rm=True,
    ):
        # TODO: find a way to be less verbose when using cached images
        if "error" in item:
            print()
            raise Exception(item["error"])
        elif "stream" in item:
            print("[dim]" + item["stream"].strip() + "[/]")
        elif "aux" in item:
            id = item["aux"].get("ID")

    if id is None:
        raise Exception("Failed to build image")

    return client.images.get(id)


def pull(image: str) -> Image:
    if client is None:
        raise Exception("Docker not available")

    return client.images.pull(image)


def run_detached(
    image: str,
    command: list[str],
    volumes: dict[str, str],
    allow_network: bool = False,
    allow_gpus: bool = True,
    allow_volume_write: bool = False,
) -> docker.models.containers.Container:
    if client is None:
        raise Exception("Docker not available")

    # by default network is disabled
    network_mode = "bridge" if allow_network else "none"

    # by default volumes are read-only
    mounts = {host: {"bind": guest, "mode": "rw" if allow_volume_write else "ro"} for host, guest in volumes.items()}

    # this allows us to log dns requests even if the container is in network mode "none"
    dns = ["127.0.0.1"] if not allow_network else None

    # enable GPUs
    device_requests = [docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])] if allow_gpus else None

    return client.containers.run(
        image,
        command=command,
        volumes=mounts,
        network_mode=network_mode,
        dns=dns,
        # automatically remove the container after it exits
        remove=True,
        # allocate a pseudo-TTY
        tty=True,
        # keep STDIN open
        stdin_open=True,
        stdout=True,
        stderr=True,
        # detach
        detach=True,
        device_requests=device_requests,
        security_opt=[
            "no-new-privileges",
            "seccomp=unconfined",
        ],
        cap_drop=["ALL"],
        tmpfs={"/tmp": "size=100m,noexec"},
        pids_limit=100,
        ulimits=[
            docker.types.Ulimit(name="nofile", soft=1024, hard=1024),
            docker.types.Ulimit(name="nproc", soft=100, hard=100),
        ],
        ipc_mode="none",
    )


def run_privileged_detached(
    image: str,
    command: list[str],
    volumes: dict[str, str],
    entrypoint: str | None = None,
    environment: dict[str, str] | None = None,
) -> docker.models.containers.Container:
    if client is None:
        raise Exception("Docker not available")

    return client.containers.run(
        image,
        command=command,
        volumes={host: {"bind": guest, "mode": "rw"} for host, guest in volumes.items()},
        network_mode="none",
        pid_mode="host",
        cgroupns="host",
        privileged=True,
        entrypoint=entrypoint,
        environment=environment,
        # automatically remove the container after it exits
        remove=True,
        # allocate a pseudo-TTY
        tty=True,
        # keep STDIN open
        stdin_open=True,
        stdout=True,
        stderr=True,
        detach=True,
    )
