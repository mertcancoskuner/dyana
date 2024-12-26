Dyana is a sandbox environment crafted for monitoring and analyzing machine learning model runtime activities. It provides detailed insights into GPU memory usage, filesystem interactions, network requests, and more during model loading and inference.

<img alt="trace" src="https://github.com/dreadnode/dyana/blob/main/examples/llama-3.2-1b-linux.png?raw=true"/>

### Model Compatibility

Dyana is designed to work with models that are compatible with [AutoModel and AutoTokenizer](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

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

Create a trace file for a given model with:

```bash
dyana trace --model /path/to/model --input "This is an example sentence." --output trace.json
```

If the model requires extra dependencies, you can pass them as:

```bash
dyana trace --model tohoku-nlp/bert-base-japanese ... --extra-requirements "protobuf fugashi ipadic"
```

**By default, Dyana will not allow network access to the model container.** If you need to allow it, you can pass the `--allow-network` flag:

```bash
dyana trace ... --allow-network
```

Show a summary of the trace file with:

```bash
dyana summary --trace trace.json
```

## TODOs

```bash
grep -r TODO .
```