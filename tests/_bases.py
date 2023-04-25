"""Shared base for embedding tests."""

from abc import ABC, abstractmethod
import sys
import unittest

from tests import _helpers


class TestEmbedBase(ABC, unittest.TestCase):
    """Abstract base to provide helpers and fixtures."""

    def setUp(self):
        _helpers.maybe_cache_embeddings_in_memory.__enter__()

        self.addCleanup(
            _helpers.maybe_cache_embeddings_in_memory.__exit__,
            *sys.exc_info(),
        )

    @property
    @abstractmethod
    def func(self):
        """Embedding function being tested."""
