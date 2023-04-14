"""Capturing ``open`` audit events for tests. This uses at most one hook."""

__all__ = ['OpenEvent', 'listening_for_open', 'skip_if_unavailable']

import contextlib
import sys
import unittest

import attrs

_hooked = False
"""Whether the audit hook has been installed."""

_open_events = None
"""The current event collector list. ``None``, whenever not collecting."""


@attrs.frozen
class OpenEvent:
    """The information from an ``open`` event that we make assertions about."""

    path = attrs.field()
    """The ``path`` argument in an ``open`` audit event."""

    mode = attrs.field()
    """The ``mode`` argument in an ``open`` audit event."""

    @classmethod
    def from_args(cls, path, mode, _):
        """Create from the actual audit event arguments. Discards ``flags``."""
        return cls(path, mode)


def _hook(event, args):
    """Auditing event hook that conditionally reports ``open`` events."""
    if event != 'open':
        return
    open_events = _open_events  # Copy reference to avoid race condition.
    if open_events is None:
        return
    open_events.append(OpenEvent.from_args(*args))


@contextlib.contextmanager
def listening_for_open():
    """Context manager to collect information on open events in a list."""
    global _hooked, _open_events

    if not _hooked:
        _hooked = True
        sys.addaudithook(_hook)

    if _open_events is not None:
        raise RuntimeError(f'{listening_for_open.__name__} is not reentrant')

    _open_events = []
    try:
        yield _open_events
    finally:
        _open_events = None


skip_if_unavailable = unittest.skipIf(
    sys.version_info < (3, 8),
    'sys.addaudithook introduced in Python 3.8',
)
"""Skip a ``unittest`` test if audit event functionality is unavailable."""
