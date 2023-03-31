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


class TestCached(unittest.TestCase):
    """Tests for disk cached versions of the embedding functions."""

    def setUp(self):
        """Create a temporary directory."""
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Delete the temporary directory."""
        self._dir.cleanup()

    # Test saving new files
    def test_saving_new_file_embed_one(self):
        prefix = 'INFO:root:embed_one: saved:'
        basename = (
            'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b'
        )
        message = f'{prefix} {self._dirname}/{basename}.json'
        with self.assertLogs() as cm:
            embed_one('hola', data_dir=self._dirname)
        self.assertEqual(cm.output, [message])

    # Test loading existing files

    # Test different functions access existing files

    # Test log corresponds to what occurred

    # Test even when data_dir is not passed

    @property
    def _dirname(self):
        return self._dir.name


if __name__ == '__main__':
    unittest.main()
