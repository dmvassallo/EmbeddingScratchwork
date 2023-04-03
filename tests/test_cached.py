#!/usr/bin/env python

"""Tests for the caching versions in the embed.cached submodule."""

# pylint: disable=missing-function-docstring

import json
import pathlib
import tempfile
import unittest

from parameterized import parameterized

from embed.cached import (
    embed_one,
    embed_many,
    embed_one_eu,
    embed_many_eu,
    embed_one_req,
    embed_many_req,
)

_HOLA_FILENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b.json'
)


class TestCached(unittest.TestCase):
    """Tests for disk cached versions of the embedding functions."""

    def setUp(self):
        """Create a temporary directory."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._dir_path = pathlib.Path(self._temporary_directory.name)

    def tearDown(self):
        """Delete the temporary directory."""
        self._temporary_directory.cleanup()

    # Test returned embeddings could plausibly be correct

    # Test saving new files
    @parameterized.expand([
        (embed_one.__name__, embed_one),
        (embed_one_eu.__name__, embed_one_eu),
        (embed_one_req.__name__, embed_one_req),
    ])
    def test_saves_file_if_not_cached(self, name, func):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:root:{name}: saved: {path}'

        with self.assertLogs() as cm:
            func('hola', data_dir=self._dir_path)

        self.assertEqual(cm.output, [expected_message])

    # Test loading existing files
    @parameterized.expand([
        (embed_one.__name__, embed_one),
        (embed_one_eu.__name__, embed_one_eu),
        (embed_one_req.__name__, embed_one_req),
    ])
    def test_loads_file_if_cached(self, name, func):
        path = self._dir_path / _HOLA_FILENAME
        expected_message = f'INFO:root:{name}: loaded: {path}'

        data = [0.0] * 1536
        with open(file=path, mode='w', encoding='utf-8') as file:
            json.dump(obj=data, fp=file)

        with self.assertLogs() as cm:
            func('hola', data_dir=self._dir_path)

        self.assertEqual(cm.output, [expected_message])

    # Test different functions access existing files
    @parameterized.expand([
        (embed_one.__name__, embed_one),
        (embed_one_eu.__name__, embed_one_eu),
        (embed_one_req.__name__, embed_one_req),
    ])
    def test_loads_from_any_implementation(self, _save_func_name, save_func):
        path = self._dir_path / _HOLA_FILENAME
        save_func('hola', data_dir=self._dir_path)

        for load_func in (embed_one, embed_one_eu, embed_one_req):
            with self.subTest(load_func=load_func):
                expected_message = (
                    f'INFO:root:{load_func.__name__}: loaded: {path}'
                )

                with self.assertLogs() as cm:
                    load_func('hola', data_dir=self._dir_path)

                self.assertEqual(cm.output, [expected_message])

    # Test log corresponds to what occurred

    # Test even when data_dir is not passed


if __name__ == '__main__':
    unittest.main()
