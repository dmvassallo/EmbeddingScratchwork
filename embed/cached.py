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
import logging
from pathlib import Path

import blake3
import numpy as np
import orjson

import embed

DEFAULT_DATA_DIR = Path('data')
"""Default directory to cache embeddings."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule."""

_serialize_embeddings = functools.partial(
    orjson.dumps,
    option=(orjson.OPT_APPEND_NEWLINE |
            orjson.OPT_INDENT_2 |
            orjson.OPT_SERIALIZE_NUMPY),
)
"""Call ``orjson.dumps`` with custom options for serializing embeddings."""


def _compute_blake3_hash(serialized_data):
    """Compute a blake3 hash of binary data."""
    # pylint: disable=not-callable  # Callable native code without type stubs.
    return blake3.blake3(serialized_data)


def _build_path(text_or_texts, data_dir):
    """Build a path for ``_disk_cache``'s wrapper to save/load embeddings."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    basename = _compute_blake3_hash(orjson.dumps(text_or_texts)).hexdigest()
    return Path(data_dir, f'{basename}.json')  # data_dir may be a str.


def _embed_with_disk_caching(func, text_or_texts, data_dir):
    """Load cached embeddings from disk, or compute and save them."""
    path = _build_path(text_or_texts, data_dir)
    try:
        json_bytes = path.read_bytes()
    except OSError:
        embeddings = func(text_or_texts)
        path.write_bytes(_serialize_embeddings(embeddings))
        _logger.info('%s: saved: %s', func.__name__, path)
    else:
        embeddings = np.array(orjson.loads(json_bytes), dtype=np.float32)
        _logger.info('%s: loaded: %s', func.__name__, path)

    return embeddings


def embed_one(text, *, data_dir=None):
    """Embed a single piece of text. Caches to disk."""
    return _embed_with_disk_caching(embed.embed_one, text, data_dir)


def embed_many(texts, *, data_dir=None):
    """Embed multiple pieces of text. Caches to disk."""
    return _embed_with_disk_caching(embed.embed_many, texts, data_dir)


def embed_one_eu(text, *, data_dir=None):
    """
    Embed a single piece of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return _embed_with_disk_caching(embed.embed_one_eu, text, data_dir)


def embed_many_eu(texts, *, data_dir=None):
    """
    Embed multiple pieces of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return _embed_with_disk_caching(embed.embed_many_eu, texts, data_dir)


def embed_one_req(text, *, data_dir=None):
    """Embed a single piece of text. Uses ``requests``. Caches to disk."""
    return _embed_with_disk_caching(embed.embed_one_req, text, data_dir)


def embed_many_req(texts, *, data_dir=None):
    """Embed multiple pieces of text. Uses ``requests``. Caches to disk."""
    return _embed_with_disk_caching(embed.embed_many_req, texts, data_dir)
