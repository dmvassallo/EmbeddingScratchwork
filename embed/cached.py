"""Versions of embedding functions that cache to disk."""

import functools
import json
import pathlib

import blake3
import numpy as np


def cache_disk(func):

    @functools.wraps(func)
    def wrapper(text_or_texts):
        # Serialize text_or_texts
        serialized = json.dumps(text_or_texts).encode()

        # Compute hash
        hash1 = blake3.blake3(serialized).hexdigest()

        path = pathlib.Path('data', f'{hash1}.json')
        try:
            with open(path, 'r', encoding='utf-8') as file:
                parsed = json.load(file)

        except OSError:
            # Get the embeddings
            embeddings = func(text_or_texts)

            # Cache the embeddings
            with open(path, 'x', encoding='utf-8') as file:
                json.dump(embeddings.tolist())

        else:
            embeddings = np.array(parsed, dtype=np.float32)

        return embeddings

    return wrapper
