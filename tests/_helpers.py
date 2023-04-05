"""Helper functions for testing."""

__all__ = [
    'getenv_bool',
    'configure_logging',
    'maybe_cache_embeddings_in_memory',
    'Caller',
]

import atexit
import contextlib
import functools
import logging
import os
import pickle
import re
import sys
import unittest.mock

import attrs

import embed

try:
    _cache_in_memory = functools.cache
except AttributeError:  # No functools.cache before Python 3.9.
    _cache_in_memory = functools.lru_cache(maxsize=None)

_truthy_config = re.compile(r'true|yes|1', re.IGNORECASE).fullmatch
"""Check if a configuration string should be considered to mean ``True``."""

_falsy_config = re.compile(r'(?:false|no|0)?', re.IGNORECASE).fullmatch
"""Check if a configuration string should be considered to mean ``False``."""


def getenv_bool(name):
    """
    Read boolean configuration from an environment variable.

    The environment variable's value is read case-insensitively, as:

    - ``True``, if it holds ``true``, ``yes``, or ``1``.
    - ``False``, if it is absent, empty, or holds ``false``, ``no``, or ``0``.
    - Otherwise, the value is  ill-formed, and ``RuntimeError`` is raised.
    """
    value = os.environ.get(name, default='')
    if _truthy_config(value):
        return True
    if _falsy_config(value):
        return False
    raise RuntimeError(
        f"Can't parse environment variable as boolean: {name}={value!r}")


@_cache_in_memory
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


def _identity_function(arg):
    """Return the argument unchanged."""
    return arg


@attrs.mutable
class _CacheStats:
    """Cache statistics (misses and hits)."""

    misses = attrs.field(default=0)
    """Number of times the cache was checked and an item was NOT found."""

    hits = attrs.field(default=0)
    """Number times the cache was checked and an an item was found."""

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
                logging.debug('In-memory cache MISS #%d: %s',
                              stats.misses, wrapper.__name__)
                value = cache[cache_key] = func(*args, **kwargs)
            else:
                stats.hits += 1
                logging.debug('In-memory cache HIT #%d: %s',
                              stats.hits, wrapper.__name__)
            return value

        return wrapper

    return decorator


_in_memory_embedding_cache_stats = _CacheStats()
"""Hits and misses of in-memory embeddings caches in tests. Not thread safe."""


@atexit.register
def _report_nontrivial_cache_statistics():
    """Log global cache statistics, IF any caching has been performed."""
    if _in_memory_embedding_cache_stats:
        logging.info('In-memory cache stats: %s',
                     _in_memory_embedding_cache_stats)


def _logged_cache_in_memory_for_testing(func):
    """Wrap an embedding function and cache, so tests make fewer API calls."""
    independent_cache = _logged_cache_in_memory_by(
        key=pickle.dumps,
        stats=_in_memory_embedding_cache_stats,
    )
    return independent_cache(func)


def _get_maybe_cache_embeddings_in_memory():
    """
    Get a decorator that may make test cases monkey-patch in in-memory caching.

    This function returns a function, which is a test fixture in decorator
    form. If tests were not configured to use in-memory caching, then
    decorating a test case with the fixture has no effect. If they were so
    configured, then decorating a test case with the fixture augments it with
    arrangement logic to monkey-patch each ``embed.embed_*`` function to equip
    it with in-memory caching, and cleanup logic to unpatch them. Although
    patching and unpatching happen on each test run, the caches live as long as
    the test runner process; the same caches, without flushing, are reused.

    In-memory caching of embeddings is not done by default. This setting is
    controlled by the ``TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY`` environment
    variable, parsed at process startup by ``getenv_bool``. When in-memory
    caching of embeddings is enabled, each embedding function uses a separate
    in-memory cache, so bugs in one are less likely to hide bugs in others.

    The fixture this function returns can be applied to a function/method or to
    a whole test class. If it is applied to a class, then the class must be a
    subclass of ``unittest.TestCase``, and the effect is the same as applying
    it to every ``test_*`` method in the class. (See ``unittest.mock.patch``.)
    """
    if not getenv_bool('TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY'):
        return _identity_function

    patches = {
        name: _logged_cache_in_memory_for_testing(getattr(embed, name))
        for name in embed.__all__
        if name.startswith('embed_')
    }
    return unittest.mock.patch.multiple(embed, **patches)


# FIXME: Probably move most info from the _get_maybe_cache_embeddings_in_memory
#        docstring to this docstring. (That's just a helper for defining this.)
maybe_cache_embeddings_in_memory = _get_maybe_cache_embeddings_in_memory()
"""Decorator that makes test case patch in in-memory caching, if enabled."""


class Caller:
    """Function proxy that supports monkey-patching module-level functions."""

    __slots__ = ('_module', '_name')

    def __init__(self, func):
        """Make a ``Caller`` for some top-level function ``func``."""
        self._module = sys.modules[func.__module__]
        self._name = func.__name__

        with contextlib.suppress(AttributeError):
            if func is self._resolve():
                return

        raise RuntimeError(f'{func!r} is not {self!r}')

    def __repr__(self):
        """Code-like representation for debugging. Possibly ``exec``-able."""
        return f'{type(self).__name__}({self._module}.{self._name})'

    def __str__(self):
        """The ``str`` of the wrapped function (as it currently resolves)."""
        return str(self._resolve())

    def __call__(self, *args, **kwargs):
        """Call the wrapped function (as it currently resolves)."""
        return self._resolve()(*args, **kwargs)

    @property
    def __name__(self):
        """The name of the (original) wrapped function from its metadata."""
        return self._name

    def _resolve(self):
        """Look up the function. If it's monkey-patched, this reflects that."""
        return getattr(self._module, self._name)
