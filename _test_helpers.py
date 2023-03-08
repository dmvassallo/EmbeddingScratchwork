"""Helper decorators for testing. This supports test_embed.py."""

# FIXME: Eliminate the need for this module, if possible, perhaps by adding a
#        dependency (or dev dependency) on a library like cachetools.
#
# TODO: If this module is kept, and the project is restructured to have a tests
#       directory, this module should go there and be renamed _helpers.py.

__all__ = ['cache_by']

import functools


def cache_by(key):
    """Like functools.cache, but uses an arbitrary key selector."""
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            try:
                return cache[cache_key]
            except KeyError:
                value = func(*args, **kwargs)
                cache[cache_key] = value
                return value

        return wrapper

    return decorator
