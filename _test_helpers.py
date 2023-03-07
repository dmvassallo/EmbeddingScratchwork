"""Helper decorators for testing. This supports test_embed.py."""

# TODO: If the project is restructured in such a way that it has a tests
#       directory, this module should go in there and be renamed _helpers.py.

__all__ = ['lazy_if', 'cache_by']

import functools

try:
    _cache = functools.cache
except AttributeError:
    _cache = functools.lru_cache(maxsize=None)


def lazy_if(condition_getter, decorator):
    """Use the decorator if a condition, computed at most once, is true."""
    def conditional_decorator(func):
        @_cache
        def maybe_decorated():
            return decorator(func) if condition_getter() else func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return maybe_decorated()(*args, **kwargs)

        return wrapper

    return conditional_decorator


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
