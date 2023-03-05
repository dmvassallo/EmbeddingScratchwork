<!-- SPDX-License-Identifier: 0BSD -->

# EmbeddingScratchwork

This is a demonstration and scratchwork repository for ways of accessing and
using OpenAI
[embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).
It uses
[text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/embedding-models),
which is [OpenAI’s second-generation embedding
model](https://openai.com/blog/new-and-improved-embedding-model).

Many embedding models, including text-embedding-ada-002, return
[normalized](https://en.wikipedia.org/wiki/Unit_vector) embeddings, whose
[cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity) are
therefore equal to their [dot
products](https://en.wikipedia.org/wiki/Dot_product). We compute similarities
with [NumPy](https://numpy.org/): individually with
[`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html),
and batched with
[`@`](https://numpy.org/doc/stable/reference/routines.linalg.html#the-operator)
([matrix
multiplication](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)).

## License

[0BSD](https://spdx.org/licenses/0BSD.html). See [**`LICENSE`**](LICENSE).

## What’s here

### Modules

[`embed.py`](embed.py) contains functions that retrieve embeddings and return
them as NumPy arrays: rank-1 arrays (vectors) for individual embeddings, or
rank-2 arrays (matrices) for batches of embeddings.

[`test_embed.py`](test_embed.py) has automated tests of the functions in
`embed.py`, including that some examples’ similarities are within expected
ranges.

### Notebooks

[`embed.ipynb`](embed.ipynb) is the main notebook. It shows some usage and
experiments, calling functions in `embed.py`.

[`structure.ipynb`](structure.ipynb) examines the JSON responses returned by
OpenAI embeddings API endpoint.

## Setup

### Way 1: Local

#### Obtaining and installing

```sh
git clone https://github.com/dmvassallo/EmbeddingScratchwork.git
cd EmbeddingScratchwork
conda env create
```

- If you fork the project, remember to replace the URL with that of your fork.
- `mamba` may be used in place of `conda` if it installed.

#### Your OpenAI API key

You need an OpenAI API key. If it is in an environment variable named
`OPENAI_API_KEY` then it will be used automatically. Otherwise you can assign
it to `openai.api_key` in your Python code.

### Way 2: GitHub Codespaces

The configuration in [`.devcontainer/`](.devcontainer/) will be used
automatically when you create a
[codespace](https://github.com/features/codespaces) on GitHub.

#### Creating the codespace

At [the repository
page](https://github.com/dmvassallo/EmbeddingScratchwork.git) (or your fork),
click the green “Code” button. Click the “Codespaces” tab and click “Create
codespace on main.”

#### Your OpenAI API key

If you want your API key to automatically be available in a codespace, you can
use your own fork and set up a repository secret.

1. [Fork the
   repository.](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
2. On the GitHub page for your fork of the repository, click “Settings.”
3. In Settings, on the left, under “Security,” expand “Secrets and variables”
   and click “Codespaces.”
4. Click “New repository secret.”
5. Put `OPENAI_API_KEY` as the name—this is so that it will appear in the
   codespace as the value of the environment variable of the same name, which
   is consulted for the OpenAI API key. As the value, put the OpenAI API key
   that you generated for the specific purpose of using for this codespace.
6. Click “Add secret.”

To expand on the point, in step 5, about using a key that is just for this,
rather than one you also use for anything else: that way, if somehow it is
accidentally disclosed, you only need to invalidate that specific key, and when
you do, none of your other projects or uses of the OpenAI API should be
affected.

### 3. Local dev container

Dev containers are often used in codespaces but you can also run them locally
with VS Code and Docker. [Here are some general
instructions](https://code.visualstudio.com/docs/devcontainers/tutorial) for
running dev containers on your own machine.

<!-- TODO: Expand the "3. Local dev container" subsection considerably. -->

## Usage

#### Activating the conda environment

To activate the conda environment in your shell:

```sh
conda activate EmbeddingScratchwork
```

Or activate the environment in an editor/IDE, such as VS Code, by selecting the
environment (or the `python` interpreter in it) in the editor/IDE’s interface.

#### Where to start

You may want to start in the [`embed.ipynb`](embed.ipynb) notebook, then look
in, adapt, and/or use the functions defined in [`embed.py`](embed.py).

## Further reading

The [OpenAI Cookbook repository](https://github.com/openai/openai-cookbook) is
the most important source of examples for using OpenAI embeddings. Example
notebooks for embeddings appear [together with other example
notebooks](https://github.com/openai/openai-cookbook/tree/main/examples).
