"""Helper functions for testing."""

__all__ = [
    'getenv_bool',
    'configure_logging',
    'cache_embeddings_in_memory',
]

import atexit
import functools
import logging
import os
import pickle
import re
import unittest.mock

import attrs

import embed

_TRUTHY_REGEX = re.compile(r'true|yes|1', re.IGNORECASE)
"""Regex to match a configuration string considered to mean ``True``."""

_FALSY_REGEX = re.compile(r'(?:false|no|0)?', re.IGNORECASE)
"""Regex to match a configuration string considered to mean ``False``."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this test helper module."""


def getenv_bool(name):
    """
    Read boolean configuration from an environment variable.

    The environment variable's value is read case-insensitively, as:

    - ``True``, if it holds ``true``, ``yes``, or ``1``.
    - ``False``, if it is absent, empty, or holds ``false``, ``no``, or ``0``.
    - Otherwise, the value is ill-formed, and ``RuntimeError`` is raised.
    """
    value = os.environ.get(name, default='')
    if _TRUTHY_REGEX.fullmatch(value):
        return True
    if _FALSY_REGEX.fullmatch(value):
        return False
    raise RuntimeError(
        f"Can't parse environment variable as boolean: {name}={value!r}")


def configure_logging():
    """
    Set logging level from the ``TESTS_LOGGING_LEVEL`` environment variable.

    If the variable is absent or empty, then no configuration is performed.
    Otherwise, the variable is treated as a string that names the logging
    level. It is case-insensitive. For example, ``DEBUG`` may be written as
    ``Debug``. Numeric and custom-defined logging levels are not supported.

    Subsequent calls to this function have no effect.
    """
    level = os.environ.get('TESTS_LOGGING_LEVEL', default='').strip().upper()
    if not level:
        return
    if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
        raise ValueError(f'unrecognized logging level {level!r}')
    logging.basicConfig(level=getattr(logging, level))


@attrs.mutable
class _CacheStats:
    """Cache statistics (misses and hits)."""

    misses = attrs.field(default=0)
    """Number of times the cache was checked and an item was NOT found."""

    hits = attrs.field(default=0)
    """Number of times the cache was checked and an item WAS found."""

    def __bool__(self):
        """Whether any cache accesses (hits or misses) have occurred."""
        return self.misses != 0 or self.hits != 0

    def __str__(self):
        """Representation for user interfaces. Doesn't show the type name."""
        return f'misses={self.misses}, hits={self.hits}'


def _logged_cache_in_memory_by(key, *, stats):
    """
    Similar to ``functools.cache``, but uses an arbitrary key selector.

    Also, this logs, is only suitable for use in tests, and is not thread-safe.
    """
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            try:
                value = cache[cache_key]
            except KeyError:
                stats.misses += 1
                _logger.debug('In-memory cache MISS #%d: %s',
                              stats.misses, wrapper.__name__)
                value = cache[cache_key] = func(*args, **kwargs)
            else:
                stats.hits += 1
                _logger.debug('In-memory cache HIT #%d: %s',
                              stats.hits, wrapper.__name__)
            return value

        return wrapper

    return decorator


_in_memory_embedding_cache_stats = _CacheStats()
"""Hits and misses of in-memory caches used in tests. Not thread safe."""


@atexit.register
def _report_nontrivial_cache_statistics():
    """Log global cache statistics, IF any caching has been performed."""
    if _in_memory_embedding_cache_stats:
        _logger.info('In-memory cache stats: %s',
                     _in_memory_embedding_cache_stats)


def _logged_cache_in_memory_for_testing(func):
    """Wrap an embedding function and cache its results in memory."""
    independent_cache = _logged_cache_in_memory_by(
        key=pickle.dumps,
        stats=_in_memory_embedding_cache_stats,
    )
    return independent_cache(func)


cache_embeddings_in_memory = unittest.mock.patch.multiple(embed, **{
    name: _logged_cache_in_memory_for_testing(getattr(embed, name))
    for name in embed.__all__
    if name.startswith('embed_')
})
"""
Arrange monkey-patching of in-memory caching for test cases.

This is a ``unittest.mock.patch`` patcher for all the ``embed.embed_*``
functions to equip them with in-memory caching, and a cleanup step that
unpatches them. The caches live as long as the test runner process, so cached
embeddings are thus reused across tests.
"""
