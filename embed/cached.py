"""Versions of embedding functions that cache to disk."""

__all__ = [
    'DEFAULT_DATA_DIR',
    'DEFAULT_FILE_TYPE',
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
import safetensors.numpy

import embed

DEFAULT_DATA_DIR = Path('data')
"""Default directory to cache embeddings."""

DEFAULT_FILE_TYPE = 'safetensors'
"""Default file type to use for caching embeddings."""

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


def _build_path(text_or_texts, data_dir, file_type):
    """Build a path for ``_disk_cache``'s wrapper to save/load embeddings."""
    data_dir = Path(DEFAULT_DATA_DIR if data_dir is None else data_dir)
    basename = _compute_input_hash(text_or_texts)
    return data_dir / f'{basename}.{file_type}'


def _embed_cache_json(func, text_or_texts, data_dir):
    """Load embeddings as JSON from disk, or compute and save them."""
    path = _build_path(text_or_texts, data_dir, 'json')
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


def _embed_cache_safetensors(func, text_or_texts, data_dir):
    """Load embeddings as safetensors from disk, or compute and save them."""
    path = _build_path(text_or_texts, data_dir, 'safetensors')
    try:
        tensor_dict = safetensors.numpy.load_file(path)
    except FileNotFoundError:
        embeddings = func(text_or_texts)
        safetensors.numpy.save_file({'embeddings': embeddings}, path)
        _logger.info('%s: saved: %s', func.__name__, path)
    else:
        embeddings = tensor_dict['embeddings']
        _logger.info('%s: loaded: %s', func.__name__, path)

    return embeddings


def _embed_cache_default(func, text_or_texts, data_dir):
    """
    Load embeddings in the default format from disk, or compute and save them.
    """
    return _CACHERS[DEFAULT_FILE_TYPE](func, text_or_texts, data_dir)


_CACHERS = {
    'json': _embed_cache_json,
    'safetensors': _embed_cache_safetensors,
    None: _embed_cache_default,
}


def embed_one(text, *, data_dir=None, file_type=None):
    """Embed a single piece of text. Caches to disk."""
    return _CACHERS[file_type](embed.embed_one, text, data_dir)


def embed_many(texts, *, data_dir=None, file_type=None):
    """Embed multiple pieces of text. Caches to disk."""
    return _CACHERS[file_type](embed.embed_many, texts, data_dir)


def embed_one_eu(text, *, data_dir=None, file_type=None):
    """
    Embed a single piece of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return _CACHERS[file_type](embed.embed_one_eu, text, data_dir)


def embed_many_eu(texts, *, data_dir=None, file_type=None):
    """
    Embed multiple pieces of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return _CACHERS[file_type](embed.embed_many_eu, texts, data_dir)


def embed_one_req(text, *, data_dir=None, file_type=None):
    """Embed a single piece of text. Uses ``requests``. Caches to disk."""
    return _CACHERS[file_type](embed.embed_one_req, text, data_dir)


def embed_many_req(texts, *, data_dir=None, file_type=None):
    """Embed multiple pieces of text. Uses ``requests``. Caches to disk."""
    return _CACHERS[file_type](embed.embed_many_req, texts, data_dir)
