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

import logging
from pathlib import Path

import blake3
import numpy as np
import orjson

import embed

DEFAULT_DATA_DIR = Path('data')
"""Default directory to cache embeddings."""

_ORJSON_SAVE_OPTIONS = (
    orjson.OPT_APPEND_NEWLINE |
    orjson.OPT_INDENT_2 |
    orjson.OPT_SERIALIZE_NUMPY
)
"""Options for ``orjson.dumps`` when it is called to serialize embeddings."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule."""


def _compute_input_hash(text_or_texts):
    """Compute a blake3-based hash of input. Used for building a filename."""
    serialized = orjson.dumps(text_or_texts)
    hasher = blake3.blake3(serialized)  # pylint: disable=not-callable
    return hasher.hexdigest()


def _build_path(text_or_texts, data_dir):
    """Build a path for ``_disk_cache``'s wrapper to save/load embeddings."""
    data_dir = Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
    basename = _compute_input_hash(text_or_texts)
    return data_dir / f'{basename}.json'


def _embed_with_disk_caching(func, text_or_texts, data_dir):
    """Load cached embeddings from disk, or compute and save them."""
    path = _build_path(text_or_texts, data_dir)
    try:
        json_bytes = path.read_bytes()
    except FileNotFoundError:
        embeddings = func(text_or_texts)
        path.write_bytes(orjson.dumps(embeddings, option=_ORJSON_SAVE_OPTIONS))
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
