"""Helper functions for testing."""

__all__ = [
    'getenv_bool',
    'configure_logging',
    'maybe_cache_embeddings_in_memory',
    'parameterize_from_suppliers',
]

import atexit
import functools
import inspect
import logging
import os
import pickle
import re
import types
import unittest.mock

from parameterized import parameterized, parameterized_class

import embed

_in_memory_embedding_cache_stats = types.SimpleNamespace(misses=0, hits=0)
"""Hits and misses of in-memory embeddings caches in tests. Not thread safe."""


@atexit.register
def _report_nontrivial_cache_statistics():
    """Log global cache statistics, IF any caching has been performed."""
    if _in_memory_embedding_cache_stats:
        logging.info('In-memory cache stats: %r',
                     _in_memory_embedding_cache_stats)


def _cache_in_memory_by(key, *, stats):
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


def _cache_in_memory_for_testing(func):
    """Wrap an embedding function and cache, so tests make fewer API calls."""
    independent_cache = _cache_in_memory_by(
        key=pickle.dumps,
        stats=_in_memory_embedding_cache_stats,
    )
    return independent_cache(func)


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
    # TODO: We have no stable interface, so drop this check after a short time.
    if 'TESTS_CACHE_EMBEDDING_CALLS' in os.environ:
        raise RuntimeError(
            'The TESTS_CACHE_EMBEDDING_CALLS environment variable is no longer'
            ' supported. Use TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY instead.',
        )

    if not getenv_bool('TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY'):
        return _identity_function

    patches = {
        name: _cache_in_memory_for_testing(getattr(embed, name))
        for name in embed.__all__
        if name.startswith('embed_')
    }
    return unittest.mock.patch.multiple(embed, **patches)


# FIXME: Probably move most info from the _get_maybe_cache_embeddings_in_memory
#        docstring to this docstring. (That's just a helper for defining this.)
maybe_cache_embeddings_in_memory = _get_maybe_cache_embeddings_in_memory()
"""Decorator that makes test case patch in in-memory caching, if enabled."""


class _IndirectCaller:
    """
    Callable object that indirectly wraps and calls a function.

    ``__call__``, ``__name__``, and ``__str__`` call the supplier each time and
    delegate to the function it returns. This extra indirection allows eager
    parameterization to work with monkey-patching.

    This facilitates using ``@parameterized.expand``/``@parameterized_class``
    together with ``@maybe_cache_embeddings_in_memory``.
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


def parameterize_from_suppliers(*func_suppliers):
    """
    Parameterize a test function or class with ``name`` and ``func``.

    With functions ``f``, ``g``, and ``h`` in modules ``a``, ``b``, and ``c``,
    use ``@parameterize_from_supplier(lambda: a.f, lambda: b.g, lambda: c.h)``.
    """
    input_values = [
        (supplier().__name__, _IndirectCaller(supplier))
        for supplier in func_suppliers
    ]
    function_decorator = parameterized.expand(input_values)
    class_decorator = parameterized_class(('name', 'func'), input_values)

    def decorator(func_or_cls):
        if inspect.isroutine(func_or_cls):
            return function_decorator(func_or_cls)
        if inspect.isclass(func_or_cls):
            return class_decorator(func_or_cls)
        raise TypeError(f"func_or_cls can't be {type(func_or_cls).__name__!r}")

    return decorator
