# Dyana

<h4 align="center">
    <a href="https://pypi.org/project/dyana/" target="_blank">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dyana">
        <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/dyana">
    </a>
    <a href="https://github.com/dreadnode/dyana/blob/main/LICENSE" target="_blank">
        <img alt="GitHub License" src="https://img.shields.io/github/license/dreadnode/dyana">
    </a>
    <a href="https://github.com/dreadnode/dyana/actions/workflows/ci.yml">
        <img alt="GitHub Actions Workflow Status" src="https://github.com/dreadnode/dyana/actions/workflows/ci.yml/badge.svg">
    </a>
</h4>

</br>

Dyana is a sandbox environment using Docker and [Tracee](https://github.com/aquasecurity/tracee) for loading, running and profiling a wide range of files, including machine learning models, ELF executables, Pickle serialized files, Javascripts [and more](https://docs.dreadnode.io/dyana/loaders/). It provides detailed insights into GPU memory usage, filesystem interactions, network requests, and security related events.

[![asciicast](https://asciinema.org/a/699347.svg)](https://asciinema.org/a/699347)

## Requirements

* Python 3.10+ with PIP.
* Docker
* Optional: a GNU/Linux machine with CUDA and the [nvidia-ctk runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU memory profiling support.

## Installation

Install with:

```bash
pip install dyana
```

To upgrade to the latest version, run:

```bash
pip install --upgrade dyana
```

To uninstall, run:

```bash
pip uninstall dyana
```

## Usage

Show a list of available loaders with:

```bash
dyana loaders
```

Show the help menu for a specific loader with:

```bash
dyana help automodel
```

Create a trace file for a given loader with:

```bash
dyana trace --loader automodel ... --output trace.json
```

To save artifacts from the container, you can pass the `--save` flag:

```bash
dyana trace --loader pip --package botocore --save /usr/local/bin/jp.py --save-to ./artifacts
```

It is possible to override the default events that Dyana will trace by passing a [custom policy](https://aquasecurity.github.io/tracee/v0.14/docs/policies/) to the tracer with:

```bash
dyana trace --loader automodel ... --policy examples/network_only_policy.yml
```

Show a summary of the trace file with:

```bash
dyana summary --trace-path trace.json
```

### Default Safeguards

Dyana does not allow network access by default to the loader container. If you need to allow it, you can pass the `--allow-network` flag:

```bash
dyana trace ... --allow-network
```

Dyana uses a shared volume to pass your files to the loader and by default it does not allow writing to it. If you need to allow it, you can pass the `--allow-volume-write` flag:

```bash
dyana trace ... --allow-volume-write
```

## Loaders

Dyana provides a set of loaders for different types of files, each loader has a dedicated set of arguments and will be executed in an isolated, offline by default container. Refer to the [documentation](https://docs.dreadnode.io/dyana/) for more information.

## License

Dyana is released under the [MIT license](LICENSE). Tracee is released under the [Apache 2.0 license](third_party_licenses/APACHE2.md).
