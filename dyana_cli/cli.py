import pathlib
import platform as platform_pkg

# NOTE: json is too slow
import cysimdjson
import typer
from rich import print

from dyana_cli.loaders.loader import Loader
from dyana_cli.tracer.tracee import Tracer
from dyana_cli.view import (
    view_disk_events,
    view_gpus,
    view_header,
    view_network_events,
    view_process_executions,
    view_ram,
    view_security_events,
)

cli = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Blackbox profiler.",
)


@cli.command(help="Profile.", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def trace(
    ctx: typer.Context,
    loader: str = typer.Option(help="Loader to use.", default="automodel"),
    platform: str | None = typer.Option(help="Platform to use.", default=None),
    output: pathlib.Path = typer.Option(help="Path to the output file.", default="trace.json"),
    timeout: int = typer.Option(help="Execution timeout in seconds.", default=60),
    no_gpu: bool = typer.Option(help="Do not use GPUs.", default=False),
    allow_network: bool = typer.Option(help="Allow network access to the model container.", default=False),
) -> None:
    # disable GPU on non-Linux systems
    if not no_gpu and platform_pkg.system() != "Linux":
        no_gpu = True

    loader = Loader(name=loader, timeout=timeout, platform=platform, args=ctx.args)
    tracer = Tracer(loader)

    trace = tracer.run_trace(allow_network, not no_gpu)

    print(f":card_file_box:  saving {len(trace.events)} events to {output}\n")

    with open(output, "w") as f:
        f.write(trace.model_dump_json())

    summary(output)


@cli.command(help="Show a summary of the trace.")
def summary(trace: pathlib.Path = typer.Option(help="Path to the trace file.", default="trace.json")) -> None:
    with open(trace) as f:
        raw = f.read()
        # the standard json parser is too slow for this
        parser = cysimdjson.JSONParser()
        trace = parser.loads(raw)

    view_header(trace)
    view_ram(trace["run"])
    view_gpus(trace["run"])

    view_process_executions(trace)
    view_network_events(trace)
    view_disk_events(trace)
    view_security_events(trace)
