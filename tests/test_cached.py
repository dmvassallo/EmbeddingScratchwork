#!/usr/bin/env python

"""Tests for the caching versions in the embed.cached submodule."""

# pylint: disable=missing-function-docstring

import tempfile
import unittest

from embed.cached import (
    embed_one,
    embed_many,
    embed_one_eu,
    embed_many_eu,
    embed_one_req,
    embed_many_req,
)


tempfile.TemporaryDirectory

class TestCached(unittest.TestCase):
    """Tests for disk cached versions of the embedding functions."""

    def setUp(self):
        """Create a temporary directory."""
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Delete the temporary directory."""
        self._dir.cleanup()

    @property
    def _dirname(self):
        return self._dir.name

    # Test saving new files

    # Test loading existing files

    # Test different functions access existing files

    # Test log corresponds to what occurred

    # Test even when data_dir is not passed

if __name__ == '__main__':
    unittest.main()
