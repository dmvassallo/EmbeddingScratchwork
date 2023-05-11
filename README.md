<!-- SPDX-License-Identifier: 0BSD -->

<!-- markdownlint-capture -->
<!-- markdownlint-disable first-line-h1 line-length no-inline-html -->
<!-- Using inline HTML to have control of image width. -->
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

### Major Modules

[`embed`](embed/__init__.py) contains functions that retrieve embeddings and
return them as NumPy arrays: rank-1 arrays (vectors) for individual embeddings,
or rank-2 arrays (matrices) for batches of embeddings.

[`embed.cached`](embed/cached.py) contains corresponding
functions that cache embeddings on disk, and check for them before contacting
OpenAI’s servers.

### Major Modules (Tests)

[`test_embed`](tests/test_embed.py) tests the functions directly in `embed`.
This includes testing that similarities are within expected ranges.

[`test_cached_embeddings`](tests/test_cached_embeddings.py) tests those same
general behaviors for the functions in `embed.cached` that cache results to
disk.

[`test_cached_caching`](tests/test_cached_caching.py) also tests the function
in `embed.cached`, but tests the on-disk caching functionality itself.

### Notebooks

[`embed.ipynb`](notebooks/embed.ipynb) is the main notebook. It shows some
usage and experiments, calling functions in the `embed` module.

[`structure.ipynb`](notebooks/structure.ipynb) examines the JSON responses
returned by the OpenAI embeddings API endpoint.

[`cached.ipynb`](notebooks/cached.ipynb) shows examples of on-disk caching and
generates some test data used in `test_cached_embeddings`.

## Setup

### Way 1: Local

Clone the project, then install it with either
[`poetry`](https://python-poetry.org/) or
[`conda`](https://en.wikipedia.org/wiki/Conda_(package_manager)).

#### Cloning the project

Wherever you want the project directory to be created, run:

```sh
git clone https://github.com/dmvassallo/EmbeddingScratchwork.git
cd EmbeddingScratchwork
```

If you forked the project (and want to use your fork), replace the URL with
that of your fork.

#### Installing with `poetry`

```sh
poetry install
```

#### Installing with `conda`

```sh
conda env create
conda activate EmbeddingScratchwork
pip install -e .
```

[`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) may be used
in place of `conda` if it is installed.

#### Your OpenAI API key

You need an [OpenAI API
key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key).
If it is in an environment variable named `OPENAI_API_KEY` then it will be used
automatically. Otherwise you can assign it to `embed.api_key` in your Python
code.

However you handle your key, make sure not to commit it to any repository. See
[Best Practices for API Key
Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

### Way 2: GitHub Codespaces

The configuration in [`.devcontainer/`](.devcontainer/) will be used
automatically when you create a
[codespace](https://github.com/features/codespaces) on GitHub.

The dev container has both the `poetry`-managed Python virtual environment and
the `conda` environment set up. So you can [activate and
use](#activating-the-environment) whichever you like.

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

#### Security of your OpenAI API key in the codespace (important)

To expand on the point, in step 5, about using a key that is just for this,
rather than one you also use for anything else: That way, if somehow it is
accidentally disclosed, you only need to invalidate that specific key. When you
do invalidate it, none of your other projects or uses of the OpenAI API should
be affected.

See [Best Practices for API Key
Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

### 3. Local dev container

Dev containers are often used in codespaces but you can also run them locally
with VS Code and Docker. The dev container has the same functionality in the
container as when run in a codespace. So, as in a codespace, the project is set
up in the container both with `poetry` and `conda` and you can use either kind
of environment.

Since a local dev container does not run in a codespace, a GitHub Codespaces
repository secret has no effect on a locally run dev container, and you’ll have
to manage your API key in some other way.

[Here are some general
instructions](https://code.visualstudio.com/docs/devcontainers/tutorial) for
running dev containers on your own machine.

<!-- TODO: Expand the "3. Local dev container" subsection considerably. -->

## Usage

### Activating the environment

#### With `poetry`

If you [installed with `poetry`](#installing-with-poetry), run this to start a
shell with the `poetry`-managed Python virtual environment activated:

```sh
poetry shell
```

#### With `conda`

If you [installed with `conda`](#installing-with-conda), run this to activate
the `conda` environment in your shell:

```sh
conda activate EmbeddingScratchwork
```

#### With your editor/IDE

Whether your installed with `poetry` or `conda`, you can activate the
environment in an editor or IDE, such as VS Code, by selecting the environment
or its Python interpreter in the editor/IDE’s interface.

For specific information about how to do this with VS Code, see [Using Python
environments in VS
Code](https://code.visualstudio.com/docs/python/environments).

### Where to start

You may want to start in the [`embed.ipynb`](notebooks/embed.ipynb) notebook.

Then look in, adapt, and/or use the functions defined in the
[`embed`](embed/__init__.py) module.

### Automated tests

There are three good ways to run the automated tests in
[`test_embed`](tests/test_embed.py):

- In a terminal, activate the environment, then run `python -m unittest`.
- In VS Code, activate the environment, then click the [beaker
  icon](https://jpearson.blog/2021/09/01/test-explorer-in-visual-studio-code/)
  and run the tests there.
- [Let CI run the tests](#continuous-integration-checks) on many combinations
  of platforms and Python versions.

## CI/CD in forks

### Continuous integration checks

This repository runs CI checks from several [GitHub Actions
workflows](.github/workflows/). Forks inherit these workflows, but they are not
usually enabled automatically. They [can be
enabled](https://github.com/github/docs/issues/15761) in a fork’s [“Actions”
tab](https://loopkit.github.io/loopdocs/gh-actions/gh-first-time/#first-use-of-actions-tab).

Once enabled, some of the workflows will run without problems. Some others—the
automated tests in [`test_embed`](tests/test_embed.py)—cannot run successfully
without an OpenAI API key.

#### Your OpenAI API key in CI checks

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
5. Put `OPENAI_API_KEY` as the name (because
   [`test.yml`](.github/workflows/test.yml) is written to expect the secret to
   have that name). As the value, put the OpenAI API key that you generated for
   the specific purpose of using for CI on this project.
6. Click “Add secret.”

#### Security of your OpenAI API key in CI checks (important)

It’s a good idea to read the relevant security guides:

- [Security hardening for GitHub
  Actions](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
  (GitHub)
- [Best Practices for API Key
  Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
  (OpenAI)

#### Deciding if `python -m unittest` should block

If you set a [configuration
variable](https://docs.github.com/en/actions/learn-github-actions/variables#creating-configuration-variables-for-a-repository)
for your fork called `TESTS_CI_NONBLOCKING`, then this determines whether CI
actions can perform multiple `python -m unittest` runs at the same time. A
value of `true` permits this. A value of `false` prohibits it; multiple test
jobs still run concurrently to download and install their dependencies, but
actually running the tests is only done by one job at a time. The default, if
you don’t set `TESTS_CI_NONBLOCKING`, is `false`.

<!-- FIXME: Change embed.py#L39-L54 link to point into embed/__init__.py. -->

This only affects CI, not [manual test runs](#automated-tests). The goal is to
make this project’s CI checks work even for users whose OpenAI accounts are
still on the trial period, whose [rate
limits](https://platform.openai.com/docs/guides/rate-limits) are [much
stricter](https://platform.openai.com/docs/guides/rate-limits/what-are-the-rate-limits-for-our-api).
All functions in this project that access the API use [exponential
backoff](https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff),
so most will succeed even if run concurrently. However, when rate limits are
very low, it is faster to run the tests in series. Furthermore, some of
them—[the
ones](https://github.com/dmvassallo/EmbeddingScratchwork/blob/4de223db30253cefba10fa6e3f846550ccd986ee/embed.py#L39-L54)
that use
[`embeddings_utils`](https://github.com/openai/openai-python/blob/v0.27.2/openai/embeddings_utils.py#L17)—do
not keep retrying enough times to reliably succeed with these much lower rate
limits.

If your OpenAI account is past the trial period, your CI checks should run
faster if you set `TESTS_CI_NONBLOCKING` to `true`.

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
