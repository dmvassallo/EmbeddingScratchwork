"""Capturing ``open`` audit events for tests. This uses at most one hook."""

__all__ = ['listening_for_open']

import contextlib
import sys

_hooked = False  # pylint: disable=invalid-name  # Not a constant.
"""Whether the audit hook has been installed."""

_listener = None  # pylint: disable=invalid-name  # Not a constant.
"""The current listener the hook passes ``open`` event args to, or ``None``."""


def _hook(event, args):
    """Auditing event hook that conditionally reports ``open`` events."""
    if event != 'open':
        return
    listener = _listener  # Copy reference to avoid race condition.
    if listener is not None:
        listener(*args)


@contextlib.contextmanager
def listening_for_open(listener):
    """Context manager to pass ``open`` event args to a listener."""
    # pylint: disable=global-statement
    # We really do want to mutate this shared state maintained at module level.
    global _hooked, _listener

    if not _hooked:
        _hooked = True
        sys.addaudithook(_hook)

    if _listener is not None:
        raise RuntimeError(f'{listening_for_open.__name__} is not reentrant')

    _listener = listener
    try:
        yield listener
    finally:
        _listener = None
