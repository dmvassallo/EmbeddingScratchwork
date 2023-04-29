"""Capturing ``open`` audit events for tests. This uses at most one hook."""

__all__ = ['listening_for_open', 'skip_if_unavailable']

import contextlib
import sys
import unittest
import unittest.mock

_hooked = False  # pylint: disable=invalid-name  # Not a constant.
"""Whether the audit hook has been installed."""

_listener_for_open = None  # pylint: disable=invalid-name  # Not a constant.
"""The current listener that ``open`` event args are passed to, or ``None``."""


def _hook(event, args):
    """Auditing event hook that conditionally reports ``open`` events."""
    if event != 'open':
        return
    listener = _listener_for_open  # Copy reference to avoid race condition.
    if _listener_for_open is not None:
        listener(*args)


@contextlib.contextmanager
def listening_for_open():
    """Context manager to pass ``open`` event args to a call-recording mock."""
    # pylint: disable=global-statement
    # We really do want to mutate this shared state maintained at module level.
    global _hooked, _listener_for_open

    if not _hooked:
        _hooked = True
        sys.addaudithook(_hook)

    if _listener_for_open is not None:
        raise RuntimeError(f'{listening_for_open.__name__} is not reentrant')

    _listener_for_open = unittest.mock.Mock()
    try:
        yield _listener_for_open
    finally:
        _listener_for_open = None


skip_if_unavailable = unittest.skipIf(
    sys.version_info < (3, 8),
    'sys.addaudithook introduced in Python 3.8',
)
"""Skip a ``unittest`` test if audit event functionality is unavailable."""
