"""Versions of embedding functions that cache to disk."""

# TODO: Maybe add a safetensors version.

__all__ = [
    'DEFAULT_DATA_DIR',
    'embed_one',
    'embed_many',
    'embed_one_eu',
    'embed_many_eu',
    'embed_one_req',
    'embed_many_req',
]

import functools
import json
import logging
import pathlib

import blake3
import numpy as np

import embed

DEFAULT_DATA_DIR = pathlib.Path('data')
"""Default directory to cache embeddings."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule."""


def _compute_blake3_hash(serialized_data):
    """Compute a blake3 hash of binary data."""
    # pylint: disable=not-callable  # Callable native code without type stubs.
    return blake3.blake3(serialized_data)


def _build_path(text_or_texts, data_dir):
    """Build a path for ``_disk_cache``'s wrapper to save/load embeddings."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    serialized_data = json.dumps(text_or_texts).encode()
    basename = _compute_blake3_hash(serialized_data).hexdigest()
    return pathlib.Path(data_dir, f'{basename}.json')  # data_dir may be a str.


def _load_json(path):
    """Load JSON from a file."""
    with open(path, mode='r', encoding='utf-8') as file:
        return json.load(file)


def _save_json(path, obj):
    """Save JSON to a file."""
    with open(path, mode='x', encoding='utf-8') as file:
        json.dump(obj=obj, fp=file, indent=4)
        file.write('\n')


def _cache_on_disk(func):
    """Decorator to add disk caching to an embedding function."""
    @functools.wraps(func)
    def wrapper(text_or_texts, *, data_dir=None):
        path = _build_path(text_or_texts, data_dir)
        try:
            parsed = _load_json(path)
        except OSError:
            embeddings = func(text_or_texts)
            _save_json(path=path, obj=embeddings.tolist())
            _logger.info('%s: saved: %s', wrapper.__name__, path)
        else:
            embeddings = np.array(parsed, dtype=np.float32)
            _logger.info('%s: loaded: %s', wrapper.__name__, path)

        return embeddings

    return wrapper


@_cache_on_disk
def embed_one(text):
    """Embed a single piece of text. Caches to disk."""
    return embed.embed_one(text)


@_cache_on_disk
def embed_many(texts):
    """Embed multiple pieces of text. Caches to disk."""
    return embed.embed_many(texts)


@_cache_on_disk
def embed_one_eu(text):
    """
    Embed a single piece of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return embed.embed_one_eu(text)


@_cache_on_disk
def embed_many_eu(texts):
    """
    Embed multiple pieces of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return embed.embed_many_eu(texts)


@_cache_on_disk
def embed_one_req(text):
    """Embed a single piece of text. Uses ``requests``. Caches to disk."""
    return embed.embed_one_req(text)


@_cache_on_disk
def embed_many_req(texts):
    """Embed multiple pieces of text. Uses ``requests``. Caches to disk."""
    return embed.embed_many_req(texts)
