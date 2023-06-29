"""Embed functions for OpenAI API experimentation."""

__all__ = [
    'cached',
    'DIMENSION',
    'TOKEN_LIMIT',
    'count_tokens',
    'embed_one',
    'embed_many',
    'embed_one_eu',
    'embed_many_eu',
    'embed_one_req',
    'embed_many_req',
]

import datetime
import http

import backoff
import numpy as np
import openai
import openai.embeddings_utils
import requests
import tiktoken

from . import _keys, cached

# Give this module an api_key property to be accessed from the outside.
_keys.initialize(__name__)

DIMENSION = 1536
"""Dimension of the vector space text-embedding-ada-002 embeds texts in."""

TOKEN_LIMIT = 8191
"""
Maximum length in cl100k_base tokens that text-embedding-ada-002 can embed.

The text passed to the ``embed_one*`` functions must not exceed this many
tokens. Each of the texts passed to the ``embed_many*`` functions must not
exceed it, though the combined length of the separate texts is not bound by it.
"""

_REQUESTS_TIMEOUT = datetime.timedelta(seconds=60)
"""Connection timeout for ``embed_one_req`` and ``embed_many_req``."""

_encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
"""An object whose ``encode`` method encodes text as cl100k_base tokens."""


def count_tokens(text):
    """Count how many tokens ``text`` is, in the model's encoding."""
    # TODO: Find a way to use less memory when counting tokens in long texts.
    return len(_encoding.encode(text))


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def _create_embedding(text_or_texts):
    """Use the OpenAI library to get one or more embeddings, with backoff."""
    return openai.Embedding.create(
        input=text_or_texts,
        model='text-embedding-ada-002',
    )


def embed_one(text):
    """Embed a single piece of text."""
    openai_response = _create_embedding(text)
    return np.array(openai_response.data[0].embedding, dtype=np.float32)


def embed_many(texts):
    """Embed multiple pieces of text."""
    openai_response = _create_embedding(texts)
    embeddings = [datum.embedding for datum in openai_response.data]
    return np.array(embeddings, dtype=np.float32)


def embed_one_eu(text):
    """Embed a single piece of text. Uses ``embeddings_utils``."""
    embedding = openai.embeddings_utils.get_embedding(
        text=text,
        engine='text-embedding-ada-002',
    )
    return np.array(embedding, dtype=np.float32)


def embed_many_eu(texts):
    """Embed multiple pieces of text. Uses ``embeddings_utils``."""
    embeddings = openai.embeddings_utils.get_embeddings(
        list_of_text=texts,
        engine='text-embedding-ada-002',
    )
    return np.array(embeddings, dtype=np.float32)


def _needs_backoff(response):
    """Check if a response has given an HTTP 429 Too Many Requests error."""
    return response.status_code == http.HTTPStatus.TOO_MANY_REQUESTS


@backoff.on_predicate(backoff.expo, _needs_backoff)
def _post_request(text_or_texts):
    """Make a POST request to the API endpoint, with backoff."""
    return requests.post(
        url='https://api.openai.com/v1/embeddings',
        headers={
            'Authorization': f'Bearer {_keys.api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'input': text_or_texts,
            'model': 'text-embedding-ada-002',
        },
        timeout=_REQUESTS_TIMEOUT.total_seconds(),
    )


def embed_one_req(text):
    """Embed a single piece of text. Uses ``requests``."""
    response = _post_request(text)
    response.raise_for_status()
    return np.array(response.json()['data'][0]['embedding'], dtype=np.float32)


def embed_many_req(texts):
    """Embed multiple pieces of text. Uses ``requests``."""
    response = _post_request(texts)
    response.raise_for_status()
    embeddings = [datum['embedding'] for datum in response.json()['data']]
    return np.array(embeddings, dtype=np.float32)
