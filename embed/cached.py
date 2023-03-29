"""Versions of embedding functions that cache to disk."""

import functools
import json
import pathlib

import blake3
import numpy as np

_DATA_DIR = pathlib.Path('data')
"""Directory to cache embeddings."""


def disk_cache(func):
    """Decorator to add disk caching to an embedding function."""
    @functools.wraps(func)
    def wrapper(text_or_texts):
        serialized = json.dumps(text_or_texts).encode()
        basename = blake3.blake3(serialized).hexdigest()
        path = _DATA_DIR / f'{basename}.json'

        try:
            with open(path, mode='r', encoding='utf-8') as file:
                parsed = json.load(file)
        except OSError:
            embeddings = func(text_or_texts)

            embeddings_list = embeddings.tolist()
            with open(path, mode='x', encoding='utf-8') as file:
                json.dump(obj=embeddings_list, fp=file)
        else:
            embeddings = np.array(parsed, dtype=np.float32)

        return embeddings

    return wrapper
