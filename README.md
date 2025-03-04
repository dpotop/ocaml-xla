# ocaml-xla
XLA (Accelerated Linear Algebra) bindings for OCaml. This is based on the
[xla-rs](https://github.com/LaurentMazare/xla-rs) Rust bindings, the semantics
for the various operands are documented on the [xla
  website](https://www.tensorflow.org/xla/operation_semantics).

![Main workflow](https://github.com/LaurentMazare/ocaml-xla/workflows/Main%20workflow/badge.svg)

Pre-compiled binaries for the xla library can be downloaded from the
[elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases/tag/v0.4.4).
These should be extracted at the root of this repository, resulting
in a `xla_extension` subdirectory being created, the currently supported version
is 0.4.4.

For a linux platform, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cpu.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cpu.tar.gz
```

For a Mac platform with M1/M2 (ARM64) hardware, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-aarch64-darwin-cpu.tar.gz
tar -xzvf xla_extension-aarch64-darwin-cpu.tar.gz
```

If the `xla_extension` directory is not in the main project directory, the path
can be specified via the `XLA_EXTENSION_DIR` environment variable.

Furthermore, on a Mac platform the path to the dynamic library must be specified by updating the environment variable ```DYLD_LIBRARY_PATH```as follows:
```
export DYLD_LIBRARY_PATH=/Users/dpotop/github/ReactiveXLA/ocaml-xla/xla_extension/lib
```


## Compilation and execution instructions for MacOS on ARM64 (M1/M2) hardware
The ```DYLD_LIBRARY_PATH```must be set to include  the ```xla_extension/lib``` folder.
Then, at the first call, execution will fail, as the dynamic library is not legalized.
To legalize it for the following calls, go to SystemPreferences/SecurityAndPrivacy 
in the General tab and click to allow its execution.

Note that by default the execution of the examples is set up on GPU.
To allow execution on Mac's ARM hardware, only the CPU code will work. 
The switch is performed in the applications themselves (the top-level
OCaML code) where variable ```use_gpu``` must be set to ```false```.

## Setting up dune for devel and debug
OCaML's build manager dune can be pretty annoying, providing very little information on the operations it performs and transforming compilation warnings into errors. For this reason, I modified its configuration in two ways:
### Global configuration
The ```config``` file has been created and must be given in argument to dune, or placed in a location I did not yet identified (everything is complicated on a mac). To give it in argument to dune:

```dune exec --config-file ./config examples/jacky.exe```

### Example compilation config (examples/dune)
I have added my new example to the file and I have modified the compilation options to avoid raising an error upon warnings, cf. https://stackoverflow.com/questions/57120927/how-to-disable-error-warning-66-unused-open-in-dune

## Generating some Text Samples with LLaMA

The [LLaMA large language model](https://github.com/facebookresearch/llama) can
be used to generate text. The model weights are only available after completing
[this form](https://forms.gle/jk851eBVbX1m5TAv5) and once downloaded can be
converted to a format this package can use. This requires a GPU with 16GB of
memory or 32GB of memory when running on CPU (tweak the `use_gpu` variable in
the example code to choose between CPU and GPU).

```bash
# Download the tokenizer config.
wget https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json -O llama-tokenizer.json

# Extract the pre-trained weights, this requires the transformers and
# safetensors python libraries to be installed.
python examples/convert_llama_checkpoint.py ..../LLaMA/7B/consolidated.00.pth

# Run the example.
dune exec examples/llama.exe
```

## Generating some Text Samples with GPT2 

One of the featured examples is GPT2. In order to run it, one should first
download the tokenization configuration file as well as the weights before
running the example. In order to do this, run the following commands:

```bash
# Download the tokenizer files.
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe

# Extract the pre-trained weights, this requires the transformers python library to be installed.
# This creates a npz file storing all the weights.
python examples/get_gpt2_weights.py

# Run the example.
dune exec examples/nanogpt.exe
```

