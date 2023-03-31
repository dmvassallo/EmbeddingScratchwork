"""Embed functions for OpenAI API experimentation."""

# TODO: Add a public submodule with versions of all 6 functions that cache (and
#       check for) embeddings on disk, possibly using safetensors.

__all__ = [
    'embed_one',
    'embed_many',
    'embed_one_eu',
    'embed_many_eu',
    'embed_one_req',
    'embed_many_req',
]

import datetime
import http
import operator

import backoff
import numpy as np
import openai
import openai.embeddings_utils
import requests

from . import _keys

# Give this module an api_key property to be accessed from the outside.
_keys.initialize(__name__)

_REQUESTS_TIMEOUT = datetime.timedelta(seconds=60)
"""Connection timeout for ``embed_one_req`` and ``embed_many_req``."""


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
    data = sorted(openai_response.data, key=operator.itemgetter('index'))
    return np.array([datum.embedding for datum in data], dtype=np.float32)


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
    """Embed a single piece of text. Use ``requests``."""
    response = _post_request(text)
    response.raise_for_status()
    return np.array(response.json()['data'][0]['embedding'], dtype=np.float32)


def embed_many_req(texts):
    """Embed multiple pieces of text. Use ``requests``."""
    response = _post_request(texts)
    response.raise_for_status()
    data = sorted(response.json()['data'], key=operator.itemgetter('index'))
    return np.array([datum['embedding'] for datum in data], dtype=np.float32)
