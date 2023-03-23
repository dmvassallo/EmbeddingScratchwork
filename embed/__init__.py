"""Embed functions for OpenAI API experimentation."""

# TODO: Add 2 more functions using requests.

# TODO: Add a public submodule with versions of all 6 functions that cache (and
#       check for) embeddings on disk, possibly using safetensors.

__all__ = [
    'embed_one',
    'embed_many',
    'embed_one_eu',
    'embed_many_eu',
]

import operator
import sys

import backoff
import numpy as np
import openai
import openai.embeddings_utils

from . import _keys

_self = sys.modules[__name__]
"""This module, to access from its own code only. (For ``_self.api_key``.)"""

# Give this module an api_key property that updates openai.api_key when set.
_self.__class__ = _keys.KeyForwardingModule

_backoff = backoff.on_exception(backoff.expo, openai.error.RateLimitError)
"""Backoff decorator for ``openai.Embedding.create``-based functions."""


@_backoff
def embed_one(text):
    """Embed a single piece of text."""
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)


@_backoff
def embed_many(texts):
    """Embed multiple pieces of text."""
    response = openai.Embedding.create(
        input=texts,
        model='text-embedding-ada-002',
    )

    data = sorted(response['data'], key=operator.itemgetter('index'))
    return np.array([datum['embedding'] for datum in data], dtype=np.float32)


def embed_one_eu(text):
    """Embed a single piece of text. Use ``embeddings_utils``."""
    embedding = openai.embeddings_utils.get_embedding(
        text=text,
        engine='text-embedding-ada-002',
    )
    return np.array(embedding, dtype=np.float32)


def embed_many_eu(texts):
    """Embed multiple pieces of text. Use ``embeddings_utils``."""
    embeddings = openai.embeddings_utils.get_embeddings(
        list_of_text=texts,
        engine='text-embedding-ada-002',
    )
    return np.array(embeddings, dtype=np.float32)
