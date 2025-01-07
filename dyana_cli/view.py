from rich import print

from dyana_cli.tracer.tracee import Tracer


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


def view_header(trace):
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

    if run["stdout"] is not None:
        print(f"[bold yellow]Stdout[/bold yellow]         : [dim]{run['stdout'][:80].strip()}[/]")

    if run["stderr"] is not None:
        print(f"[bold red]Stderr[/bold red]         : {run['stderr'][:80].strip()}")

    if run["exit_code"] is not None:
        print(f"[bold blue]Exit code[/bold blue]      : {run['exit_code']}")

    print()


def view_ram(run):
    ram = run["ram"]
    if ram:
        print("[bold yellow]RAM:[/]")
        ram_stages = list(ram.keys())
        prev_stage = None
        for stage in ram_stages:
            if prev_stage is None:
                print(f"  * {stage} : {sizeof_fmt(ram[stage])}")
            else:
                print(f"  * {stage} : {delta_fmt(ram[prev_stage], ram[stage])}")
            prev_stage = stage

        print()


def view_gpus(run):
    if run["gpu"]:
        gpu_stages = list(run["gpu"].keys())
        first_gpu_stage = gpu_stages[0]
        num_gpus = len(run["gpu"][first_gpu_stage])
        if num_gpus:
            print("[bold green]GPU:[/]")

            for i in range(num_gpus):
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


def view_process_executions(trace):
    proc_execs = [event for event in trace["events"] if event["eventName"] == "sched_process_exec"]
    if proc_execs:
        print("[bold yellow]Process Executions:[/]")
        for proc_exec in proc_execs:
            cmd_path = [arg["value"] for arg in proc_exec["args"] if arg["name"] == "cmdpath"][0]
            cmd_argv = [list(arg["value"]) for arg in proc_exec["args"] if arg["name"] == "argv"][0]
            print(f"  * {proc_exec['processName']} -> [bold red]{proc_exec['syscall']}[/] {cmd_path} {cmd_argv}")
        print()


def view_network_events(trace):
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
                answers = [f'{a["name"]}={a["IP"]}' for a in data["answers"]]

                if not answers:
                    print(f"  * {event['processName']} | [bold red]dns[/] | question={', '.join(question_names)}")
                else:
                    print(f"  * {event['processName']} | [bold red]dns[/] | answer={', '.join(answers)}")

        print()


def view_disk_events(trace):
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


def view_security_events(trace):
    security_events = {event for event in trace["events"] if event["eventName"] in Tracer.SECURITY_EVENTS}
    if security_events:
        print("[bold red]Security Events:[/]")
        unique = {event["metadata"]["Properties"]["signatureName"]: event["metadata"] for event in security_events}
        for signature, event in unique.items():
            category = event["Properties"]["Category"]
            severity_level = event["Properties"]["Severity"]
            print(f"  * {signature} ([dim]{category}[/], {severity_fmt(severity_level)})")
        print()
