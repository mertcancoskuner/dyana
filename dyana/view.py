import typing as t

from rich import box, print
from rich.table import Table

from dyana.loaders.loader import Loader
from dyana.tracer.tracee import Tracer


def _view_loader_help_markdown(loader: Loader) -> None:
    print(f"# {loader.name}\n")
    print(f"{loader.settings.description}")
    print()
    print("* **Requires Network:**", "yes" if loader.settings.network else "no")
    if loader.settings.build_args:
        print("* **Optional Build Arguments:**", ", ".join({f"`--{k}`" for k in loader.settings.build_args.keys()}))

    if loader.settings.args:
        print()
        print("## Arguments")
        print()

        print(
            "| Argument     | Description                                                         | Default                      | Required |"
        )
        print(
            "|--------------|---------------------------------------------------------------------|------------------------------|----------|"
        )
        for arg in loader.settings.args:
            print(f"| `--{arg.name}` | {arg.description} | `{arg.default}` | {'yes' if arg.required else 'no'} |")

    if loader.settings.examples:
        print()
        print("## Examples")
        print()
        for example in loader.settings.examples:
            print(f"{example.description}\n")
            print(f"```bash\n{example.command}\n```")
            print()


def view_loader_help(loader: Loader, markdown: bool) -> None:
    if loader.settings:
        if markdown:
            _view_loader_help_markdown(loader)
        else:
            print(f"[bold green]{loader.name}[/] - {loader.settings.description}\n")
            if loader.settings.network:
                print("Network    : [bold red]yes[/]")
            else:
                print("Network    : [dim]no[/]")

            if loader.settings.build_args:
                print("Build args :", ", ".join({f"[yellow]--{k}[/]" for k in loader.settings.build_args.keys()}))

            if loader.settings.args:
                print("")
                table = Table(box=box.ROUNDED)
                table.add_column("Argument", style="yellow")
                table.add_column("Description")
                table.add_column("Default")
                table.add_column("Required")

                for arg in loader.settings.args:
                    table.add_row(
                        f"--{arg.name}",
                        arg.description,
                        f"[dim]{arg.default}[/]" if arg.default else "",
                        str(arg.required),
                    )
                print(table)

            if loader.settings.examples:
                print()
                print("[bold]Examples[/]")
                print()
                for example in loader.settings.examples:
                    print(f"{example.description}\n")
                    print(f"  [dim]{example.command}[/]")
                    print()


# https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
def sizeof_fmt(num: float, suffix: str = "B") -> str:
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


def view_header(trace: dict[str, t.Any]) -> None:
    run = trace["run"]

    print(f"Platform       : [magenta]{trace['platform']}[/]")
    print(f"Loader         : [bold]{trace['run']['loader_name']}[/]")

    if run["build_args"]:
        print(f"Build args     : {', '.join(f'{k}={v}' for k, v in run['build_args'].items())}")

    if run["arguments"]:
        print(f"Arguments      : {' '.join(run['arguments'])}")

    if run["volumes"]:
        print(f"Volumes        : {', '.join(f'{v} ({k})' for k, v in run['volumes'].items())}")

    print(f"Started at     : {trace['started_at']}")
    print(f"Ended at       : {trace['ended_at']}")
    print(f"Total Events   : {len(trace['events'])}")

    if run["errors"]:
        print()
        print("[bold red]Errors:[/bold red]\n")
        for group, error in run["errors"].items():
            if error:
                print(f"  * [b]{group}[/]: {error}")
        print()

    if run["warnings"]:
        print()
        print("[bold yellow]Warnings:[/bold yellow]\n")
        for group, warning in run["warnings"].items():
            if warning:
                print(f"  * [b]{group}[/]: {warning}")
        print()

    if run["stdout"]:
        print(f"[bold yellow]Stdout[/bold yellow]         : [dim]{run['stdout'].strip()}[/]")

    if run["stderr"]:
        print(f"[bold red]Stderr[/bold red]         : {run['stderr'].strip()}")

    if run["exit_code"]:
        print(f"[bold blue]Exit code[/bold blue]      : {run['exit_code']}")

    print()


def view_ram(run: dict[str, t.Any]) -> None:
    ram = run["ram"]
    if ram:
        print("[bold yellow]RAM Usage:[/]")
        ram_stages = list(ram.keys())
        prev_stage = None
        for stage in ram_stages:
            if prev_stage is None:
                print(f"  * {stage} : {sizeof_fmt(ram[stage])}")
            else:
                print(f"  * {stage} : {delta_fmt(ram[prev_stage], ram[stage])}")
            prev_stage = stage

        print()


def view_gpus(run: dict[str, t.Any]) -> None:
    if run["gpu"]:
        gpu_stages = list(run["gpu"].keys())
        first_gpu_stage = gpu_stages[0]
        num_gpus = len(run["gpu"][first_gpu_stage])
        if num_gpus:
            # check for any change in memory usage for GPUs
            changes = []
            for i in range(num_gpus):
                prev = None
                change = False
                for stage in gpu_stages:
                    if prev is not None:
                        if run["gpu"][stage][i]["free_memory"] != prev:
                            change = True
                            break
                    prev = run["gpu"][stage][i]["free_memory"]
                changes.append(change)

            if any(changes):
                print("[bold green]GPU Usage:[/]")
                for i in range(num_gpus):
                    if not changes[i]:
                        continue

                    dev_name = run["gpu"][first_gpu_stage][i]["device_name"]
                    dev_total = run["gpu"][first_gpu_stage][i]["total_memory"]

                    print(f"  [green]{dev_name}[/] [dim]|[/] {sizeof_fmt(dev_total)}")

                    prev_stage = None
                    for stage in gpu_stages:
                        used = run["gpu"][stage][i]["total_memory"] - run["gpu"][stage][i]["free_memory"]
                        if prev_stage is None:
                            print(f"  * {stage} : {sizeof_fmt(used)}")
                        else:
                            print(f"  * {stage} : {delta_fmt(prev_stage, used)}")
                        prev_stage = used

                    print()


def view_disk_usage(run: dict[str, t.Any]) -> None:
    if "disk" in run and run["disk"]:
        print("[bold yellow]Disk Usage:[/]")
        disk = run["disk"]
        disk_stages = list(disk.keys())
        prev_stage = None
        for stage in disk_stages:
            if prev_stage is None:
                print(f"  * {stage} : {sizeof_fmt(disk[stage])}")
            else:
                print(f"  * {stage} : {delta_fmt(disk[prev_stage], disk[stage])}")
            prev_stage = stage

        print()


def view_exec_tree(exec: dict[str, t.Any], level: int = 0) -> None:
    pad = "  " * level
    print(f"{pad}* [dim]{exec['processId']}[/] {exec['command']}")
    for child in exec["children"]:
        view_exec_tree(child, level + 1)


def view_process_executions(trace: dict[str, t.Any]) -> None:
    proc_execs = [event for event in trace["events"] if event["eventName"] == "sched_process_exec"]
    if proc_execs:
        print("[bold yellow]Process Executions:[/]")

        execs = {}
        for proc_exec in proc_execs:
            cmd_path = [arg["value"] for arg in proc_exec["args"] if arg["name"] == "cmdpath"][0]
            cmd_argv = [list(arg["value"]) for arg in proc_exec["args"] if arg["name"] == "argv"][0]
            execs[proc_exec["processId"]] = {
                "processId": proc_exec["processId"],
                "parentProcessId": proc_exec["parentProcessId"],
                "command": f"{proc_exec['processName']} -> [bold red]{proc_exec['syscall']}[/] {cmd_path} {cmd_argv}",
                "children": [],
            }

        tree = []
        for _, exec in execs.items():
            parent_pid = exec["parentProcessId"]
            if parent_pid in execs:
                execs[parent_pid]["children"].append(exec)
            else:
                tree.append(exec)

        for exec in tree:
            view_exec_tree(exec)

        print()


def view_network_usage(run: dict[str, t.Any]) -> None:
    has_network_usage = "network" in run and run["network"]
    if has_network_usage:
        network = run["network"]
        stages = list(network.keys())
        interfaces = list(network[stages[0]].keys())
        any_change = False

        for interface in interfaces:
            for stage in stages:
                if network[stage][interface]["rx"] > 0 or network[stage][interface]["tx"] > 0:
                    any_change = True
                    break

        if any_change:
            print("[bold yellow]Network Usage:[/]")

            for interface in interfaces:
                # Check if there were any network changes across stages
                had_network_activity = False
                for stage in stages:
                    if network[stage][interface]["rx"] > 0 or network[stage][interface]["tx"] > 0:
                        had_network_activity = True
                        break

                if not had_network_activity:
                    continue

                print(f"  [bold]{interface}[/]")
                prev_stage = None
                for stage in stages:
                    if prev_stage is None:
                        rx_fmt = sizeof_fmt(network[stage][interface]["rx"])
                        tx_fmt = sizeof_fmt(network[stage][interface]["tx"])
                        print(f"    {stage} : rx={rx_fmt} tx={tx_fmt}")
                    else:
                        rx_fmt = delta_fmt(network[prev_stage][interface]["rx"], network[stage][interface]["rx"])
                        tx_fmt = delta_fmt(network[prev_stage][interface]["tx"], network[stage][interface]["tx"])
                        print(f"    {stage} : rx={rx_fmt} tx={tx_fmt}")
                    prev_stage = stage

                print()


def view_network_events(trace: dict[str, t.Any]) -> None:
    connects = [event for event in trace["events"] if event["eventName"] == "security_socket_connect"]
    dns_queries = [event for event in trace["events"] if event["eventName"] == "net_packet_dns"]
    if connects or dns_queries:
        print("[bold yellow]Network Activity:[/]")

        all = connects + dns_queries
        all.sort(key=lambda e: e["timestamp"])

        visualized = []

        for event in all:
            if event["eventName"] == "security_socket_connect":
                remote_addr = [arg["value"] for arg in event["args"] if arg["name"] == "remote_addr"][0]
                endpoint = "?"
                if remote_addr and "sa_family" in remote_addr:
                    family = remote_addr["sa_family"]
                    if family == "AF_UNIX":
                        endpoint = remote_addr["sun_path"]
                    elif family == "AF_INET":
                        endpoint = f"{remote_addr['sin_addr']}:{remote_addr['sin_port']}"
                    elif family == "AF_INET6":
                        endpoint = f"[{remote_addr['sin6_addr']}]:{remote_addr['sin6_port']}"

                line = f"  * [[dim]{event['processId']}[/]] {event['processName']} -> [bold red]{event['syscall']}[/] {endpoint}"

                if line not in visualized:
                    visualized.append(line)
                    print(line)

            else:
                data = [arg["value"] for arg in event["args"] if arg["name"] == "proto_dns"][0]
                question_names = [q["name"] for q in data["questions"]]
                answers = [f'{a["name"]}={a["IP"]}' for a in data["answers"]]

                if not answers:
                    line = f"  * [[dim]{event['processId']}[/]] {event['processName']} | [bold red]dns[/] | question={', '.join(question_names)}"
                else:
                    line = f"  * [[dim]{event['processId']}[/]] {event['processName']} | [bold red]dns[/] | answer={', '.join(answers)}"

                if line not in visualized:
                    visualized.append(line)
                    print(line)

        print()


def view_disk_events(trace: dict[str, t.Any]) -> None:
    opens = [event for event in trace["events"] if event["eventName"] == "security_file_open"]
    unique_files = set()
    any_file = False
    any_special = False
    special_paths: dict[str, int] = {
        "/usr/local/lib/": 0,
        "/app/node_modules/": 0,
        "/usr/lib/": 0,
        "/lib/": 0,
        "/dev/": 0,
        "/proc/": 0,
        "/sys/": 0,
        "/etc/": 0,
        "/usr/share/": 0,
        "/tmp/": 0,
        "/var/": 0,
    }

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


def view_security_events(trace: dict[str, t.Any]) -> None:
    security_events = {event for event in trace["events"] if event["eventName"] in Tracer.SECURITY_EVENTS}
    if security_events:
        print("[bold yellow]Security Events:[/]")

        unique = {}
        for event in security_events:
            if "metadata" in event:
                unique[event["metadata"]["Properties"]["signatureName"]] = event["metadata"]
            else:
                unique[event["eventName"]] = event

        for signature, event in unique.items():
            if "Properties" in event:
                category = event["Properties"]["Category"]
                severity_level = event["Properties"]["Severity"]
            else:
                category = "misc"
                severity_level = 0

            print(f"  * {signature} ([dim]{category}[/], {severity_fmt(severity_level)})")

        print()


def view_extra_unknown(key: str, value: t.Any) -> None:
    if value:
        print(f"[bold yellow]{key.title()}:[/] ")
        print(f"  {value}")
        print()


def count_package_prefixes(path_dict: dict[str, str], level: int = 2) -> dict[str, int]:
    from collections import defaultdict

    prefix_counter: defaultdict[str, int] = defaultdict(int)

    for package_path in path_dict.keys():
        parts = package_path.split(".")
        if len(parts) >= level:
            prefix = ".".join(parts[:level])
        else:
            prefix = parts[0]

        prefix_counter[prefix] += 1

    return dict(prefix_counter)


def view_extra_imports(key: str, value: t.Any) -> None:
    if value:
        print("[bold yellow]Top Level Imports:[/] ")
        as_dict = dict(value.items())
        as_counters = count_package_prefixes(as_dict, level=1)
        for package, count in sorted(as_counters.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                print(f"  * [green]{package}[/][dim].*[/]: {count}")
            else:
                print(f"  * [green]{package}[/]")
        print()


def view_extra(run: dict[str, t.Any]) -> None:
    unknown = []
    if "extra" in run and run["extra"]:
        for k, v in run["extra"].items():
            fn_name = f"view_extra_{k}"
            if fn_name in globals():
                globals()[fn_name](k, v)
            else:
                unknown.append(k)

    if unknown:
        print("[bold yellow]Other Records:[/]")
        for k in unknown:
            print(f"  * {k}")
        print()
