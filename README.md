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

</br>

Dyana is a sandbox environment using Docker and [Tracee](https://github.com/aquasecurity/tracee) for loading, running and profiling a wide range of files, including machine learning models, ELF executables, Pickle serialized files, Javascripts and more. It provides detailed insights into GPU memory usage, filesystem interactions, network requests, and security related events.

## Installation

Requirements:

* Python 3.10+ with PIP.
* Docker
* Optional: a GNU/Linux machine with CUDA for GPU memory tracing support.

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

Create a trace file for a given loader with:

```bash
dyana trace --loader automodel ... --output trace.json
```

**By default, Dyana will not allow network access to the model container.** If you need to allow it, you can pass the `--allow-network` flag:

```bash
dyana trace ... --allow-network
```

Show a summary of the trace file with:

```bash
dyana summary --trace-path trace.json
```

## Loaders

Dyana provides a set of loaders for different types of files, each loader has a dedicated set of arguments and will be executed in an isolated, offline by default container.

To see the available loaders and their scriptions, run `dyana loaders`.

### automodel

The default loader for machine learning models. It will load any model that is compatible with [AutoModel and AutoTokenizer](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

#### Example Usage

```bash
dyana trace --loader automodel --model /path/to/model --input "This is an example sentence."

# automodel is the default loader, so this is equivalent to:
dyana trace --model /path/to/model --input "This is an example sentence."


# in case the model requires extra dependencies, you can pass them as:
dyana trace --model tohoku-nlp/bert-base-japanese --input "This is an example sentence." --extra-requirements "protobuf fugashi ipadic"
```

<img alt="automodel" src="https://github.com/dreadnode/dyana/blob/main/examples/llama-3.2-1b-linux.png?raw=true"/>

### elf

This loader will load an ELF file and run it.

#### Example Usage

```bash
dyana trace --loader elf --elf /path/to/linux_executable

# depending on the ELF file and the host computer, you might need to specify a different platform:
dyana trace --loader elf --elf /path/to/linux_executable --platform linux/amd64

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader elf --elf /path/to/linux_executable --allow-network
```

<img alt="elf" src="https://github.com/dreadnode/dyana/blob/main/examples/mirai.png?raw=true"/>

### pickle

This loader will load a Pickle serialized file.

#### Example Usage

```bash
dyana trace --loader pickle --pickle /path/to/file.pickle

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader pickle --pickle /path/to/file.pickle --allow-network
```

![pickle](./examples/malicious-pickle-on-macos.png)

### python

This loader will load a Python file and run it.

#### Example Usage

```bash
dyana trace --loader python --script /path/to/file.py

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader python --script /path/to/file.py --allow-network
```

![python](./examples/python-hello-on-macos.png)

### js

This loader will load a Javascript file and run it via NodeJS.

#### Example Usage

```bash
dyana trace --loader js --script /path/to/file.js

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader js --script /path/to/file.js --allow-network
```

![js](./examples/js-hello-on-macos.png)

## License

Dyana is released under the [MIT license](LICENSE). Tracee is released under the [Apache 2.0 license](third_party_licenses/APACHE2.md).