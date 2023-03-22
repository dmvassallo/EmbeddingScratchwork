<!-- SPDX-License-Identifier: 0BSD -->

<!-- Logo. Tell markdownlint it's OK this precedes <h1> and has long lines. -->
<!-- markdownlint-capture -->
<!-- markdownlint-disable MD041 MD013 -->
<img src="doc/logo.svg"
     alt="Drawing of text-embedding-ada-002 embedding vectors for two sentences – “El gato corre.” and “The cat runs.” – and the 22.7° angle between them"
     title="Drawing of text-embedding-ada-002 embedding vectors for two sentences – “El gato corre.” and “The cat runs.” – and the 22.7° angle between them"
     width="500px">
<!-- markdownlint-restore -->

# EmbeddingScratchwork

This is a demonstration and scratchwork repository for ways of accessing and
using OpenAI
[embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).
It uses
[text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/embedding-models),
which is [OpenAI’s second-generation embedding
model](https://openai.com/blog/new-and-improved-embedding-model).

Many embedding models, [including all of
OpenAI’s](https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use),
return [normalized](https://en.wikipedia.org/wiki/Unit_vector) embeddings,
whose [cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity)
are therefore equal to their [dot
products](https://en.wikipedia.org/wiki/Dot_product). We compute similarities
with [NumPy](https://numpy.org/): individually with
[`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), or
batched with
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
`embed.py`. This includes testing that some examples’ similarities are within
expected ranges.

### Notebooks

[`embed.ipynb`](embed.ipynb) is the main notebook. It shows some usage and
experiments, calling functions in `embed.py`.

[`structure.ipynb`](structure.ipynb) examines the JSON responses returned by
the OpenAI embeddings API endpoint.

## Setup

### Way 1: Local

#### Obtaining and installing

Clone the repository and create its
[`conda`](https://en.wikipedia.org/wiki/Conda_(package_manager)) environment:

```sh
git clone https://github.com/dmvassallo/EmbeddingScratchwork.git
cd EmbeddingScratchwork
conda env create
```

- If you fork the project, remember to replace the URL with that of your fork.
- [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) may be
  used in place of `conda` if it is installed.

#### Your OpenAI API key

You need an [OpenAI API
key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key).
If it is in an environment variable named `OPENAI_API_KEY` then it will be used
automatically. Otherwise you can assign it to `openai.api_key` in your Python
code.

However you handle your key, make sure not to commit it to any repository. See
[Best Practices for API Key
Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

### Way 2: GitHub Codespaces

The configuration in [`.devcontainer/`](.devcontainer/) will be used
automatically when you create a
[codespace](https://github.com/features/codespaces) on GitHub.

#### Creating the codespace

At [the repository
page](https://github.com/dmvassallo/EmbeddingScratchwork.git) (or your fork),
click the green “Code” button. Click the “Codespaces” tab and click “Create
codespace on main.”

#### Your OpenAI API key in the codespace

If you want your [API
key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)
to automatically be available in a codespace, you can use your own fork and set
up a repository secret. *Of course, do not commit your key to your repository.*

1. [Fork the
   repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   if you haven’t already, and go to your fork on GitHub.
2. Click the “Settings” tab. (You want the settings for your repository, not
   for your whole GitHub account.)
3. In Settings, on the left, under “Security,” expand “Secrets and variables”
   and click “**Codespaces**.”
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
affected. See [Best Practices for API Key
Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

### 3. Local dev container

Dev containers are often used in codespaces but you can also run them locally
with VS Code and Docker. [Here are some general
instructions](https://code.visualstudio.com/docs/devcontainers/tutorial) for
running dev containers on your own machine.

<!-- TODO: Expand the "3. Local dev container" subsection considerably. -->

## Usage

### Activating the conda environment

To activate the conda environment in your shell:

```sh
conda activate EmbeddingScratchwork
```

Or activate the environment in an editor/IDE, such as VS Code, by selecting the
environment (or the `python` interpreter in it) in the editor/IDE’s interface.

For specific information about how to do this with VS Code, see [Using Python
environments in VS
Code](https://code.visualstudio.com/docs/python/environments).

### Where to start

You may want to start in the [`embed.ipynb`](embed.ipynb) notebook, then look
in, adapt, and/or use the functions defined in [`embed.py`](embed.py).

## CI/CD in forks

### Continuous integration checks

This repository defines CI checks in several [GitHub Actions
workflows](.github/workflows/). Forks inherit them. Some will run without
problems. Some others—the automated tests in
[`test_embed.py`](test_embed.py)—cannot run successfully without an OpenAI API
key.

Since your API key must *not* be committed or otherwise disclosed, the way to
make it available to CI (if you wish to do so) is by setting up a [repository
secret for GitHub
actions](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository).
This differs from any repository secret you may have created for your fork’s
*codespaces*, because it is a repository secret for *actions* rather than
codespaces:

1. Go to your fork on GitHub.
2. Click the “Settings” tab. (You want the settings for your repository, not
   for your whole GitHub account.)
3. In Settings, on the left, under “Security,” expand “Secrets and variables”
   and click “**Actions**.”
4. Click “New repository secret.”
5. Put `OPENAI_API_KEY` as the name—this is so that it will appear in the
   codespace as the value of the environment variable of the same name, which
   is consulted for the OpenAI API key. As the value, put the OpenAI API key
   that you generated for the specific purpose of using for this codespace.
6. Click “Add secret.”

It’s a good idea to read the relevant security guides:

- [Security hardening for GitHub
  Actions](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
  (GitHub)
- [Best Practices for API Key
Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
(OpenAI)

### Codespace prebuilds

Currently, a dev container for EmbeddingScratchwork takes several minutes to
create because it downloads and installs development tools and dependencies. To
do this ahead of time instead of each time you create a codespace, you can
[configure
prebuilds](https://docs.github.com/en/codespaces/prebuilding-your-codespaces/configuring-prebuilds).
As of this writing, this repository has prebuilds configured. We do *not*
guarantee that this will remain the case. But the bigger issue is that, if you
want prebuilds for your *fork*, you must set them up separately.

You may want to trigger prebuilds on a schedule, such as once per day, rather
than on each push. This is because the most common situation where a
prebuild—of this particular repository—becomes out of date is when there are
updates to the development tools that are installed in the dev container, or to
the project’s dependencies.

Prebuilds use GitHub Codespaces
[storage](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/viewing-your-github-codespaces-usage),
which can bring you closer to the monthly cap and, if you pay for additional
Codespaces usage/storage, can cause you to incur greater costs than otherwise.
When configuring a prebuild, you can decrease this effect by setting “Template
history” to 1 instead of the default of 2. Especially if you made your fork
mainly for personal experimentation or to open pull requests on this
repository—rather than to take the software in a substantially different
direction—you may well consider your prebuild important only for your own use.
In that case, reducing its “Region availability” to only your region would be a
good way to decrease your Codespaces storage even further.

We have found it helpful to [disable prebuild
optimization](https://docs.github.com/en/codespaces/troubleshooting/troubleshooting-prebuilds#preventing-out-of-date-prebuilds-being-used)
for prebuilds on this repository.

## Further reading

The [OpenAI Cookbook repository](https://github.com/openai/openai-cookbook) is
the most important source of examples for using OpenAI embeddings. Example
notebooks for embeddings appear [together with other example
notebooks](https://github.com/openai/openai-cookbook/tree/main/examples).
