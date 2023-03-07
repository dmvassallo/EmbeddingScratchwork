"""Embed functions for OpenAI API experimentation."""

# TODO: Add 4 more functions: 2 using embeddings_utils, and 2 using requests.

__all__ = ['embed_one', 'embed_many']

import operator

import backoff
import numpy as np
import openai

_backoff = backoff.on_exception(backoff.expo, openai.error.RateLimitError)


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
