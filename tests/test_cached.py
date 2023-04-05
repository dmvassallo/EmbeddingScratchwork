#!/usr/bin/env python

"""
Tests for embedding functions in the ``embed.cached`` submodule.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

import json
import pathlib
import tempfile
from typing import Any
import unittest

from parameterized import parameterized_class

from embed import cached
from tests import _helpers

_helpers.configure_logging()

_HOLA_FILENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b.json'
)
"""Filename that would be generated from the input ``'hola'``."""


@parameterized_class(('name', 'func'), [
    (cached.embed_one.__name__, staticmethod(cached.embed_one)),
    (cached.embed_one_eu.__name__, staticmethod(cached.embed_one_eu)),
    (cached.embed_one_req.__name__, staticmethod(cached.embed_one_req)),
])
@_helpers.maybe_cache_embeddings_in_memory
class TestDiskCachedEmbedOne(unittest.TestCase):
    """Tests of ``embed.cached.embed_one*`` functions, which cache to disk."""

    name: Any
    func: Any

    def setUp(self):
        """Create a temporary directory."""
        # pylint: disable=consider-using-with  # tearDown cleans this up.
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._dir_path = pathlib.Path(self._temporary_directory.name)

    def tearDown(self):
        """Delete the temporary directory."""
        self._temporary_directory.cleanup()

    # Test returned embeddings could plausibly be correct

    # Test saving new files
    def test_saves_file_if_not_cached(self):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:root:{self.name}: saved: {path}'

        with self.assertLogs() as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    # Test loading existing files
    def test_loads_file_if_cached(self):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:root:{self.name}: loaded: {path}'

        data = [0.0] * 1536
        with open(file=path, mode='w', encoding='utf-8') as file:
            json.dump(obj=data, fp=file)

        with self.assertLogs() as log_context:
            self.func('hola', data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    # Test different functions access existing files
    def test_saved_file_can_load_with_any_implementation(self):
        path = self._dir_path / _HOLA_FILENAME
        self.func('hola', data_dir=self._dir_path)

        for load_func in (cached.embed_one,
                          cached.embed_one_eu,
                          cached.embed_one_req):
            with self.subTest(load_func=load_func):
                expected_message = (
                    f'INFO:root:{load_func.__name__}: loaded: {path}'
                )

                with self.assertLogs() as log_context:
                    load_func('hola', data_dir=self._dir_path)

                self.assertEqual(log_context.output, [expected_message])

    # Test log corresponds to what occurred

    # Test even when data_dir is not passed


# FIXME: Test the embed.cached.embed_many* functions.


if __name__ == '__main__':
    unittest.main()
