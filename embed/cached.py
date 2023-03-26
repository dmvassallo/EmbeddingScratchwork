"""Versions of embedding functions that cache to disk."""

import blake3
import functools


def cache_disk(func):

    @functools.wraps(func)
    def wrapper(text_or_texts):

        # Compute hash

        # Is hash already a key

        # NO: Call the function

        # NO: Store the key and embedding/s

        # Return embedding/s
        return ...

    return wrapper
