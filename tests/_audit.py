"""Capturing audit events for tests. This uses at most one hook."""

__all__ = ['extract', 'skip_if_unavailable']

import contextlib
import sys
import threading
import unittest


_lock = threading.Lock()
"""Mutex protecting the table."""

_table = None
"""Table mapping each event to its listeners, or None if not yet needed."""


def _hook(event, args):
    """Single audit hook used for all events and handlers."""
    try:
        # For performance, don't lock. Subscripting a dict with str keys should
        # be sufficiently protected by the GIL in CPython. This doesn't protect
        # the table rows. But those are tuples that we always replace, rather
        # than lists that we mutate, so we should observe consistent state.
        listeners = _table[event]
    except KeyError:
        return

    for listener in listeners:
        listener(*args)


def _subscribe(event, listener):
    """Attach a detachable listener to an event."""
    global _table

    with _lock:
        if _table is None:
            _table = {}
            sys.addaudithook(_hook)

        old_listeners = _table.get(event, ())
        _table[event] = (*old_listeners, listener)


def _fail_unsubscribe(event, listener):
    """Raise an exception for an unsuccessful attempt to detach a listener."""
    raise ValueError(f'{event!r} listener {listener!r} never subscribed')


def _unsubscribe(event, listener):
    """Detach a listener that was attached to an event."""
    with _lock:
        if _table is None:
            _fail_unsubscribe(event, listener)

        try:
            listeners = _table[event]
        except KeyError:
            _fail_unsubscribe(event, listener)

        # Work with the sequence in reverse to remove the most recent listener.
        listeners_reversed = list(reversed(listeners))
        try:
            listeners_reversed.remove(listener)
        except ValueError:
            _fail_unsubscribe(event, listener)

        if listeners_reversed:
            _table[event] = tuple(reversed(listeners_reversed))
        else:
            del _table[event]


@contextlib.contextmanager
def _listen(event, listener):
    """Context manager that subscribes and unsubscribes an event listener."""
    _subscribe(event, listener)
    try:
        yield
    finally:
        _unsubscribe(event, listener)


@contextlib.contextmanager
def extract(event, extractor):
    """Context manager that provides a list of custom-extracted even data."""
    extracts = []
    with _listen(event, lambda *args: extracts.append(extractor(*args))):
        yield extracts


skip_if_unavailable = unittest.skipIf(
    sys.version_info < (3, 8),
    'sys.addaudithook introduced in Python 3.8',
)
"""Skip a ``unittest`` test if audit event functionality is unavailable."""
