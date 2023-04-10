"""Capturing audit events for tests. This uses at most one hook."""

__all__ = ['scoped_listener']

import contextlib
import sys
import threading


_lock = threading.Lock()
"""Mutex protecting the table."""

_table = None
"""Table mapping each event to its listeners, or None if not yet needed."""


def _hook(event, args):
    """Single audit hook used for all events and handlers."""
    for listener in _table.get(event, default=()):
        listener(*args)


def _subscribe(event, listener):
    """Attach a detachable listener to an event."""
    global _table

    with _lock:
        if _table is None:
            _table = {}
            sys.addaudithook(_hook)
        old_listeners = _table.get(event, default=())
        _table[event] = (*old_listeners, listener)


def _unsubscribe_raise(event, listener):
    """Raise an exception for an unsuccessful attempt to detach a listener."""
    raise ValueError(f'{event!r} listener {listener!r} never subscribed')


def _unsubscribe(event, listener):
    """Detach a listener that was attached to an event."""
    with _lock:
        if _table is None or (listeners := _table.get(event)) is None:
            _unsubscribe_raise(event, listener)

        # Work with the sequence in reverse to remove the most recent listener.
        listeners_reversed = list(reversed(listeners))
        try:
            listeners_reversed.remove(listener)
        except ValueError:
            _unsubscribe_raise(event, listener)

        if listeners_reversed:
            _table[event] = tuple(reversed(listeners_reversed))
        else:
            del _table[event]


@contextlib.contextmanager
def scoped_listener(event, listener):
    """Context manager that subscribes and unsubscribes an event listener."""
    _subscribe(event, listener)
    try:
        yield
    finally:
        _unsubscribe(event, listener)
