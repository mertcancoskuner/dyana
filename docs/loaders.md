Dyana provides a set of loaders for different types of files, each loader has a dedicated set of arguments and will be executed in an isolated, offline by default container.

To see the available loaders and their scriptions, run `dyana loaders`.

### automodel

The default loader for machine learning models. It will load any model that is compatible with [AutoModel and AutoTokenizer](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html).

```bash
dyana trace --loader automodel --model /path/to/model --input "This is an example sentence."

# automodel is the default loader, so this is equivalent to:
dyana trace --model /path/to/model --input "This is an example sentence."


# in case the model requires extra dependencies, you can pass them as:
dyana trace --model tohoku-nlp/bert-base-japanese --input "This is an example sentence." --extra-requirements "protobuf fugashi ipadic"
```

![Automodel Loader](assets/loader-automodel.png)

### elf

This loader will load an ELF file and run it.

```bash
dyana trace --loader elf --elf /path/to/linux_executable

# depending on the ELF file and the host computer, you might need to specify a different platform:
dyana trace --loader elf --elf /path/to/linux_executable --platform linux/amd64

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader elf --elf /path/to/linux_executable --allow-network
```

![ELF Loader](assets/loader-elf.png)

### pickle

This loader will load a Pickle serialized file.

```bash
dyana trace --loader pickle --pickle /path/to/file.pickle

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader pickle --pickle /path/to/file.pickle --allow-network
```

![Pickle Loader](assets/loader-pickle.png)

### python

This loader will load a Python file and run it.

```bash
dyana trace --loader python --script /path/to/file.py

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader python --script /path/to/file.py --allow-network
```

![Python Loader](assets/loader-python.png)

### js

This loader will load a Javascript file and run it via NodeJS.

```bash
dyana trace --loader js --script /path/to/file.js

# networking is disabled by default, if you need to allow it, you can pass the --allow-network flag:
dyana trace --loader js --script /path/to/file.js --allow-network
```

![JS Loader](assets/loader-js.png)