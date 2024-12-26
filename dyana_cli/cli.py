import pathlib
import platform

# NOTE: json is too slow
import cysimdjson
import typer
from rich import print

from dyana_cli.loaders.loader import Loader
from dyana_cli.tracer.tracee import Tracer

cli = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Blackbox model profiler.",
)


@cli.command(help="Profile a model.")
def trace(
    model: pathlib.Path = typer.Option(help="Path to the model to profile."),
    loader: str = typer.Option(help="Loader to use.", default="automodel"),
    input: str = typer.Option(help="Input for the model.", default="This is an example sentence."),
    output: pathlib.Path = typer.Option(help="Path to the output file.", default="trace.json"),
    no_gpu: bool = typer.Option(help="Do not use GPUs.", default=False),
    allow_network: bool = typer.Option(help="Allow network access to the model container.", default=False),
    extra_requirements: str | None = typer.Option(help="Extra requirements to install.", default=None),
) -> None:
    # disable GPU on non-Linux systems
    if not no_gpu and platform.system() != "Linux":
        no_gpu = True

    allow_gpus = not no_gpu
    build_args = {"EXTRA_REQUIREMENTS": extra_requirements} if extra_requirements else None

    # TODO: for now we only have "auto", figure out more specific loaders
    loader = Loader(loader, build_args)
    tracer = Tracer(loader)

    trace = tracer.run_trace(model, input, allow_network, allow_gpus)

    if trace.errors:
        print(":exclamation: [bold red]errors:[/bold red]")
        for group, errors in trace.errors.items():
            for error in errors:
                print(f"  [b]{group}[/]: {error}")
        print()

    print(f":card_file_box:  saving {len(trace.events)} events to {output}\n")

    with open(output, "w") as f:
        f.write(trace.model_dump_json())

    summary(output)


# https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def delta_fmt(before: int, after: int) -> str:
    delta = after - before
    fmt = sizeof_fmt(after)
    if delta > 0:
        delta_fmt = sizeof_fmt(delta)
        fmt += f" :red_triangle_pointed_up: [red]{delta_fmt}[/]"
    return fmt


def severity_fmt(level: int) -> str:
    if level >= 3:
        return "[bold red]high severity[/]"
    elif level >= 2:
        return "[bold yellow]moderate severity[/]"
    elif level >= 1:
        return "[bold green]low severity[/]"
    else:
        return "[bold dim]no severity[/]"


@cli.command(help="Show a summary of the trace.")
def summary(trace: pathlib.Path = typer.Option(help="Path to the trace file.", default="trace.json")) -> None:
    with open(trace) as f:
        raw = f.read()
        # the standard json parser is too slow for this
        parser = cysimdjson.JSONParser()
        trace = parser.loads(raw)

    # TODO: make this prettier

    print()

    ram = trace["ram"]

    tot_mem_pressure = max(
        ram["start"],
        ram["after_tokenizer_loaded"],
        ram["after_tokenization"],
        ram["after_model_loaded"],
        ram["after_model_inference"],
    )

    tot_gpu_pressure: int = 0
    num_gpus: int = 0
    if "gpu" in trace and trace["gpu"]:
        num_gpus = len(trace["gpu"]["start"])
        for i in range(num_gpus):
            usage = (
                trace["gpu"]["after_model_inference"][i]["total_memory"]
                - trace["gpu"]["after_model_inference"][i]["free_memory"]
            )
            tot_gpu_pressure += usage

    print(f"Platform       : [magenta]{trace['platform']}[/]")
    if "extra_requirements" in trace:
        print(f"Extra packages : {trace['extra_requirements']}")
    print(f"Model path     : [yellow]{trace['model_path']}[/]")
    print(f"Model input    : [dim]{trace['model_input']}[/]")
    print(f"Started at     : {trace['started_at']}")
    print(f"Ended at       : {trace['ended_at']}")
    print(f"RAM usage      : [yellow][bold]{sizeof_fmt(tot_mem_pressure)}[/]")
    if tot_gpu_pressure:
        print(f"GPU vRAM usage : [green][bold]{sizeof_fmt(tot_gpu_pressure)}[/]")
    print(f"Total Events   : {len(trace['events'])}")

    print()

    if trace["errors"]:
        print("[bold red]Errors:[/bold red]\n")
        for group, errors in trace["errors"].items():
            for error in errors:
                print(f"  * [b]{group}[/]: {error}")
        print()

    print("[bold yellow]RAM:[/]")
    print(f"  * start            : {sizeof_fmt(ram['start'])}")
    print(f"  * tokenizer loaded : {delta_fmt(ram['start'], ram['after_tokenizer_loaded'])}")
    print(f"  * tokenization     : {delta_fmt(ram['after_tokenizer_loaded'], ram['after_tokenization'])}")
    print(f"  * model loaded     : {delta_fmt(ram['after_tokenization'], ram['after_model_loaded'])}")
    print(f"  * model inference  : {delta_fmt(ram['after_model_loaded'], ram['after_model_inference'])}")
    print()

    if "gpu" in trace and num_gpus:
        print("[bold green]GPU:[/]")

        for i in range(num_gpus):
            dev_name = trace["gpu"]["start"][i]["device_name"]
            dev_total = trace["gpu"]["start"][i]["total_memory"]
            dev_free = trace["gpu"]["start"][i]["free_memory"]
            at_start = dev_total - dev_free

            try:
                after_tokenizer_loaded = (
                    trace["gpu"]["after_tokenizer_loaded"][i]["total_memory"]
                    - trace["gpu"]["after_tokenizer_loaded"][i]["free_memory"]
                )
            except IndexError:
                after_tokenizer_loaded = 0

            try:
                after_tokenization = (
                    trace["gpu"]["after_tokenization"][i]["total_memory"]
                    - trace["gpu"]["after_tokenization"][i]["free_memory"]
                )
            except IndexError:
                after_tokenization = 0

            try:
                after_model_loaded = (
                    trace["gpu"]["after_model_loaded"][i]["total_memory"]
                    - trace["gpu"]["after_model_loaded"][i]["free_memory"]
                )
            except IndexError:
                after_model_loaded = 0

            try:
                after_model_inference = (
                    trace["gpu"]["after_model_inference"][i]["total_memory"]
                    - trace["gpu"]["after_model_inference"][i]["free_memory"]
                )
            except IndexError:
                after_model_inference = 0

            # skip this GPU if it was not influenced by the trace
            if (
                len(
                    {
                        at_start,
                        after_tokenizer_loaded,
                        after_tokenization,
                        after_model_loaded,
                        after_model_inference,
                    }
                )
                > 1
            ):
                print(f"  [green]{dev_name}[/] [dim]|[/] {sizeof_fmt(dev_total)}")
                print(f"    * start           : {sizeof_fmt(at_start)}")
                print(f"    * tokenizer loaded: {delta_fmt(at_start, after_tokenizer_loaded)}")
                print(f"    * tokenization    : {delta_fmt(after_tokenizer_loaded, after_tokenization)}")
                print(f"    * model loaded    : {delta_fmt(after_tokenization, after_model_loaded)}")
                print(f"    * model inference : {delta_fmt(after_model_loaded, after_model_inference)}")
                print()

    proc_execs = [event for event in trace["events"] if event["eventName"] == "sched_process_exec"]
    if proc_execs:
        print("[bold yellow]Process Executions:[/]")
        for proc_exec in proc_execs:
            cmd_path = [arg["value"] for arg in proc_exec["args"] if arg["name"] == "cmdpath"][0]
            cmd_argv = [[v for v in arg["value"]] for arg in proc_exec["args"] if arg["name"] == "argv"][0]
            print(f"  * {proc_exec['processName']} -> [bold red]{proc_exec['syscall']}[/] {cmd_path} {cmd_argv}")
        print()

    connects = [event for event in trace["events"] if event["eventName"] == "security_socket_connect"]
    dns_queries = [event for event in trace["events"] if event["eventName"] == "net_packet_dns"]
    if connects or dns_queries:
        print("[bold yellow]Network:[/]")

        all = connects + dns_queries
        all.sort(key=lambda e: e["timestamp"])

        for event in all:
            if event["eventName"] == "security_socket_connect":
                remote_addr = [arg["value"] for arg in event["args"] if arg["name"] == "remote_addr"][0]
                remote_addr_family = remote_addr["sa_family"]
                remote_addr_fields = [f"{k}={v}" for k, v in remote_addr.items() if k != "sa_family"]

                print(
                    f"  * {event['processName']} -> [bold red]{event['syscall']}[/] {remote_addr_family} {', '.join(remote_addr_fields)}"
                )

            else:
                data = [arg["value"] for arg in event["args"] if arg["name"] == "proto_dns"][0]
                question_names = [q["name"] for q in data["questions"]]
                answers = [f'{a["name"]}={a['IP']}' for a in data["answers"]]

                if not answers:
                    print(f"  * {event['processName']} | [bold red]dns[/] | question={', '.join(question_names)}")
                else:
                    print(f"  * {event['processName']} | [bold red]dns[/] | answer={', '.join(answers)}")

        print()

    opens = [event for event in trace["events"] if event["eventName"] == "security_file_open"]
    unique_files = set()
    any_file = False
    any_special = False
    special_paths: dict[str, int] = {
        "/usr/local/lib/": 0,
        "/usr/lib/": 0,
        "/lib/": 0,
        "/dev/": 0,
        "/proc/": 0,
        "/sys/": 0,
        "/etc/": 0,
    }

    # py_packages = {}

    for file in opens:
        file_path = [arg["value"] for arg in file["args"] if arg["name"] == "syscall_pathname"][0]
        if not file_path:
            file_path = [arg["value"] for arg in file["args"] if arg["name"] == "pathname"][0]

        is_special_path = False
        any_file = True

        for special_path in special_paths:
            if special_path in file_path:
                special_paths[special_path] += 1
                is_special_path = True
                any_special = True
                break

        if not is_special_path:
            unique_files.add(file_path)

        """
        TODO: ! WIP ! generalize this type of stuff os a policy, collect SBoM for other loaders

        if "/python" in file_path and "/site-packages/" in file_path:
            # Extract the first folder after site-packages/
            package_name = file_path.split("/site-packages/")[1].split("/")[0]
            package_version = None

            if package_name.endswith(".dist-info"):
                parts = package_name.split("-", 1)
                package_name = parts[0]
                package_version = parts[1].replace(".dist-info", "")

            if package_name != "__pycache__":
                if package_name not in py_packages:
                    py_packages[package_name] = package_version
        """

    if any_file:
        print("[bold yellow]File Accesses:[/]")
        for file_path in sorted(unique_files):
            print(f"  * {file_path}")

        if any_special:
            print()
            for path, count in special_paths.items():
                if count > 0:
                    print(f"  * {count} accesses to {path}[dim]*[/]")

        print()

    # print(py_packages)

    security_events = {event for event in trace["events"] if event["eventName"] in Tracer.SECURITY_EVENTS}
    if security_events:
        print("[bold red]Security Events:[/]")
        unique = {event["metadata"]["Properties"]["signatureName"]: event["metadata"] for event in security_events}
        for signature, event in unique.items():
            category = event["Properties"]["Category"]
            severity_level = event["Properties"]["Severity"]
            print(f"  * {signature} ([dim]{category}[/], {severity_fmt(severity_level)})")
        print()
