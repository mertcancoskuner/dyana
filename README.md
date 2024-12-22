Dyana is a sandbox environment designed for loading and running machine learning models, offering comprehensive tracing of runtime activities such as detailed memory usage (including both host and GPU), filesystem interactions, network requests, and more.

Models are loaded within a containerized environment, followed by an inference execution. The runtime tracing is highly lightweight, utilizing eBPF to provide in-depth monitoring and analysis.

<img alt="trace" src="https://github.com/dreadnode/dyana/blob/main/trace.png?raw=true"/>

### Model Compatibility

Dyana is designed to work with models that are compatible with [AutoModel and AutoTokenizer](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

## Requirements

* Docker
* [Poetry](https://python-poetry.org/)
* Optional: a GNU/Linux machine with CUDA for GPU tracing support.

## Usage

Activate the Poetry shell:

```bash
cd /path/to/dyana
poetry install && poetry shell
```

Create a trace file for a given model with:

```bash
dyana trace --model /path/to/model --input "This is an example sentence." --output trace.json
```

Show a summary of the trace file with:

```bash
dyana summary --trace trace.json
```

## TODOs

```bash
grep -r TODO .
```