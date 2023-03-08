"""Helper decorators for testing. This supports test_embed.py."""

# TODO: If this module is kept, and the project is restructured to have a tests
#       directory, then this module should go there and be renamed _helpers.py.

__all__ = ['configure_logging', 'get_maybe_caching_decorator']

import functools
import logging
import os
import pickle
import re

_cache_hits = _cache_misses = 0


def configure_logging():
    """
    Set logging level from the ``TESTS_LOGGING_LEVEL`` environment variable.

    If the variable is absent or empty, then no configuration is performed.
    Otherwise, the variable is treated as a string that names the logging
    level. It is case-insensitive. For example, ``DEBUG`` may be written as
    ``Debug``. Numeric and custom-defined logging levels are not supported.
    """
    level = os.environ.get('TESTS_LOGGING_LEVEL', default='').strip().upper()
    if not level:
        return
    if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
        raise ValueError(f'unrecognized logging level {level!r}')
    logging.basicConfig(level=getattr(logging, level))


def get_maybe_caching_decorator():
    """
    Get a decorator to use on unary functions that may or may not add caching.

    The decision of whether to cache or not is made eagerly, in this function.

    If the ``TESTS_CACHE_EMBEDDING_CALLS`` environment variable exists and
    holds ``yes`` or ``true`` (case insensitively) or any positive integer, the
    returned decorator caches. Pickling is used for cache keys, so non-hashable
    arguments are supported, and arguments of different types are treated as
    different, even if equal.

    Otherwise, the returned decorator is just an identity function.
    """
    if re.match(
        pattern=r'\A\s*(?:yes|true|\+?0*[1-9][0-9]*)\s*\Z',
        string=os.environ.get('TESTS_CACHE_EMBEDDING_CALLS', default=''),
        flags=re.IGNORECASE,
    ):
        return _cache_by(pickle.dumps)

    return lambda func: func


def _cache_by(key):
    """
    Like ``functools.cache``, but uses an arbitrary key selector.

    This also logs, and globally counts cache hits and misses.
    """
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # FIXME: Count in a way that avoids using global state.
            global _cache_hits, _cache_misses

            cache_key = key(*args, **kwargs)
            try:
                value = cache[cache_key]
            except KeyError:
                _cache_misses += 1
                logging.debug('Cache MISS (global count: %d): %s',
                              _cache_misses, wrapper.__name__)
                value = cache[cache_key] = func(*args, **kwargs)
            else:
                _cache_hits += 1
                logging.debug('Cache HIT (global count: %d): %s',
                              _cache_hits, wrapper.__name__)
            return value

        return wrapper

    return decorator
