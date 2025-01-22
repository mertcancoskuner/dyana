<p align="center">
    <img
    src="https://d1lppblt9t2x15.cloudfront.net/logos/5714928f3cdc09503751580cffbe8d02.png"
    alt="Logo"
    align="center"
    width="144px"
    height="144px"
    />
</p>

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

<p align="center">
    <strong>
        <a href="https://docs.dreadnode.io/dyana/" target="_blank">
            Documentation
        </a>
    </strong>
</p>

</br>

Dyana is a sandbox environment using Docker and [Tracee](https://github.com/aquasecurity/tracee) for loading, running and profiling a wide range of files, including machine learning models, ELF executables, Pickle serialized files, Javascripts [and more](https://docs.dreadnode.io/dyana/loaders/). It provides detailed insights into GPU memory usage, filesystem interactions, network requests, and security related events.

## Requirements

* Python 3.10+ with PIP.
* Docker
* Optional: a GNU/Linux machine with CUDA for GPU memory tracing support.

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

Dyana provides a set of loaders for different types of files, each loader has a dedicated set of arguments and will be executed in an isolated, offline by default container. Click on a loader to see the arguments and examples.

<details>
<summary><b>automodel</b></summary>

The default loader for machine learning models. It will load any model that is compatible with [AutoModel and AutoTokenizer](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

```bash
dyana trace --loader automodel --model /path/to/model --input "This is an example sentence."

# automodel is the default loader, so this is equivalent to:
dyana trace --model /path/to/model --input "This is an example sentence."

# in case the model requires extra dependencies, you can pass them as:
dyana trace --model tohoku-nlp/bert-base-japanese --input "This is an example sentence." --extra-requirements "protobuf fugashi ipadic"

# to perform tokenizer and model initialization without loading weights.
dyana trace --model /path/to/model --low-memory
```
</details>

<details>
<summary><b>elf</b></summary>
This loader will load an ELF file and run it.

```bash
dyana trace --loader elf --elf /path/to/linux_executable

# depending on the ELF file and the host computer, you might need to specify a different platform:
dyana trace --loader elf --elf /path/to/linux_executable --platform linux/amd64

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader elf --elf /path/to/linux_executable --allow-network
```
</details>

<details>
<summary><b>pickle</b></summary>
This loader will load a Pickle serialized file.

```bash
dyana trace --loader pickle --pickle /path/to/file.pickle

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader pickle --pickle /path/to/file.pickle --allow-network
```
</details>

<details>
<summary><b>python</b></summary>
This loader will load a Python file and run it.

```bash
dyana trace --loader python --script /path/to/file.py

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader python --script /path/to/file.py --allow-network
```
</details>

<details>
<summary><b>pip</b></summary>
This loader will install a Python package via PIP.

```bash
dyana trace --loader pip --package requests

# you can install a specific version of a package:
dyana trace --loader pip --package requests==2.28.2

# you can also pass extra dependencies to be installed:
dyana trace --loader pip --package foobar --extra-dependencies "gcc"
```
</details>

<details>
<summary><b>js</b></summary>
This loader will load a Javascript file and run it via NodeJS.

```bash
dyana trace --loader js --script /path/to/file.js

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader js --script /path/to/file.js --allow-network
```
</details>

<details>
<summary><b>npm</b></summary>
This loader will install a Javascript package via NPM.

```bash
dyana trace --loader npm --package express

# you can install a specific version of a package:
dyana trace --loader npm --package express@1.0.0

# you can also pass extra dependencies to be installed:
dyana trace --loader npm --package express --extra-dependencies "axios"
```

</details>

<details>
<summary><b>website</b></summary>
Opens a website in a headless browser and profiles its performance.

```bash
dyana trace --loader website --url https://www.google.com

# you can also save a screenshot of the page
dyana trace --loader website --url https://www.google.com --screenshot

# you can also wait for a specific element to load
dyana trace --loader website --url https://www.google.com --wait-for "body"
```

</details>

## License

Dyana is released under the [MIT license](LICENSE). Tracee is released under the [Apache 2.0 license](third_party_licenses/APACHE2.md).
