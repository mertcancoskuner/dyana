Dyana is a sandbox environment based on Docker and [eBPF](https://github.com/aquasecurity/tracee) crafted for loading, running and monitoring machine learning models, ELF files and more. It provides detailed insights into GPU memory usage, filesystem interactions, network requests, and more.

## Loaders

Dyana provides a set of loaders for different types of files, each loader has a dedicated set of arguments.

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

### run_elf

This loader will load an ELF file and run it.

#### Example Usage

```bash
dyana trace --loader run_elf --elf /path/to/linux_executable

# depending on the ELF file and the host computer, you might need to specify a different platform:
dyana trace --loader run_elf --elf /path/to/linux_executable --platform linux/amd64

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader run_elf --elf /path/to/linux_executable --allow-network
```

<img alt="run_elf" src="https://github.com/dreadnode/dyana/blob/main/examples/linux-exe-on-macos.png?raw=true"/>


## Requirements

* Docker
* [Poetry](https://python-poetry.org/)
* Optional: a GNU/Linux machine with CUDA for GPU memory tracing support.

## Usage

Activate the Poetry shell:

```bash
cd /path/to/dyana
poetry install && poetry shell
```

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
dyana summary --trace trace.json
```