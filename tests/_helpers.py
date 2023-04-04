"""Helper functions for testing."""

__all__ = [
    'getenv_bool',
    'configure_logging',
    'get_maybe_cache_in_memory_decorator',
    'IndirectCaller',
]

import atexit
import collections
import functools
import logging
import os
import pickle
import re

_in_memory_cache_stats = collections.Counter()
"""Mapping that tracks global cache hit and miss counts. Not thread safe."""


@atexit.register
def _report_nontrivial_cache_statistics():
    """Log global cache statistics, IF any caching has been performed."""
    if _in_memory_cache_stats:
        logging.info('In-memory cache stats: %r', _in_memory_cache_stats)


def _cache_in_memory_by(key, *, stats):
    """
    Similar to ``functools.cache``, but uses an arbitrary key selector.

    This logs, is only suitable for use in tests, and is not thread-safe.
    """
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            try:
                value = cache[cache_key]
            except KeyError:
                stats['misses'] += 1
                logging.debug('In-memory cache MISS #%d: %s',
                              stats['misses'], wrapper.__name__)
                value = cache[cache_key] = func(*args, **kwargs)
            else:
                stats['hits'] += 1
                logging.debug('In-memory cache HIT #%d: %s',
                              stats['hits'], wrapper.__name__)
            return value

        return wrapper

    return decorator


def _identity_function(arg):
    """Return the argument unchanged."""
    return arg


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


# FIXME: Somehow rename TESTS_CACHE_EMBEDDING_CALLS (everywhere) to something
#        that clarifies it is not specifically related to embed.cached caching.
def get_maybe_cache_in_memory_decorator():
    """
    Get a decorator to use on unary functions that might add in-memory caching.

    The decision of whether to cache or not is made eagerly, in this function.

    If the ``TESTS_CACHE_EMBEDDING_CALLS`` environment variable holds a truthy
    value (``true``, ``yes``, or ``1`, case-insensitively), the returned
    decorator caches in memory. Pickling is used for cache keys, to support
    non-hashable arguments and treat arguments of different types as different
    even if equal. The in-memory cache is only suitable for use in tests (so
    they make fewer API calls on CI). In particular, it isn't thread-safe.

    Otherwise, the returned decorator is just an identity function.
    """
    if getenv_bool('TESTS_CACHE_EMBEDDING_CALLS'):
        return _cache_in_memory_by(pickle.dumps, stats=_in_memory_cache_stats)
    return _identity_function


# FIXME: Document the purpose of this class in greater detail.
class IndirectCaller:
    """
    Callable object that indirectly wraps and calls a function.

    The indirection allows eager parameterization to work with monkey-patching.

    ``__call__``, ``__name__``, and ``__str__`` delegate to the function.
    """

    __slots__ = ('_supplier',)

    def __init__(self, func_supplier):
        """Make a caller from a function supplier. Pass ``lambda: func``."""
        self._supplier = func_supplier

    def __repr__(self):
        """Vaguely code-like representation for debugging."""
        return f'{type(self).__name__}(lambda: {self._supplier()!r})'

    def __str__(self):
        """Fetch the function from the supplier and convert it to a string."""
        return str(self._supplier())

    def __call__(self, *args, **kwargs):
        """Fetch the function from the supplier and call it."""
        return self._supplier()(*args, **kwargs)

    @property
    def __name__(self):
        """The name of the function returned by the supplier."""
        return self._supplier().__name__
