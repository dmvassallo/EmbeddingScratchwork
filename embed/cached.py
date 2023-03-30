"""Versions of embedding functions that cache to disk."""

__all__ = [
    'embed_one',
    'embed_many',
    'embed_one_eu',
    'embed_many_eu',
    'embed_one_req',
    'embed_many_req',
]

import functools
import json
import pathlib

import blake3
import numpy as np

import embed


def _disk_cache(func):
    """Decorator to add disk caching to an embedding function."""
    @functools.wraps(func)
    def wrapper(text_or_texts, *, data_dir='data'):
        serialized = json.dumps(text_or_texts).encode()
        basename = blake3.blake3(serialized).hexdigest()
        path = pathlib.Path(data_dir, f'{basename}.json')

        try:
            with open(path, mode='r', encoding='utf-8') as file:
                parsed = json.load(file)
        except OSError:
            embeddings = func(text_or_texts)

            embeddings_list = embeddings.tolist()
            with open(path, mode='x', encoding='utf-8') as file:
                json.dump(obj=embeddings_list, fp=file, indent=4)
                file.write('\n')
        else:
            embeddings = np.array(parsed, dtype=np.float32)

        return embeddings

    return wrapper


@_disk_cache
def embed_one(text):
    """Embed a single piece of text. Caches to disk."""
    return embed.embed_one(text)


@_disk_cache
def embed_many(texts):
    """Embed multiple pieces of text. Caches to disk."""
    return embed.embed_many(texts)


@_disk_cache
def embed_one_eu(text):
    """
    Embed a single piece of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return embed.embed_one_eu(text)


@_disk_cache
def embed_many_eu(texts):
    """
    Embed multiple pieces of text. Uses ``embeddings_utils``. Caches to disk.
    """
    return embed.embed_many_eu(texts)


@_disk_cache
def embed_one_req(text):
    """Embed a single piece of text. Uses ``requests``. Caches to disk."""
    return embed.embed_one_req(text)


@_disk_cache
def embed_many_req(texts):
    """Embed multiple pieces of text. Uses ``requests``. Caches to disk."""
    return embed.embed_many_req(texts)
