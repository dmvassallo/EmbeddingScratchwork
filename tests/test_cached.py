#!/usr/bin/env python

"""
Tests specific to the embedding functions in the ``embed.cached`` submodule.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

from abc import abstractmethod
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import embed
from embed import cached
from tests import _audit, _bases, _helpers

_HOLA_FILENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b.json'
)
"""Filename that would be generated from the input ``'hola'``."""

_HOLA_HELLO_FILENAME = (
    '4a77f419587b08963e94105b8b9272531e53ade9621b613fda175aa0a96cd839.json'
)
"""Filename that would be generated from the input ``['hola', 'hello']``."""

_helpers.configure_logging()


class _TestDiskCachedBase(_bases.TestEmbedBase):
    """Shared test fixture logic for all tests of disk caching versions."""

    def setUp(self):
        """Create a temporary directory."""
        super().setUp()

        # pylint: disable=consider-using-with  # tearDown cleans this up.
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._dir_path = pathlib.Path(self._temporary_directory.name)

    def tearDown(self):
        """Delete the temporary directory."""
        self._temporary_directory.cleanup()
        super().tearDown()


class _TestDiskCachedCachingBase(_TestDiskCachedBase):
    """
    Tests specific to caching behavior of ``embed.cached.embed*`` functions.
    """

    @property
    @abstractmethod
    def text_or_texts(self):
        """Input to the embedding functions."""

    @property
    @abstractmethod
    def filename(self):
        """
        Filename that should be generated from the input ``text_or_texts``.
        """

    @property
    @abstractmethod
    def fake_data(self):
        """Fake data for testing loads from a file."""

    @property
    @abstractmethod
    def func_group(self):
        """
        Single or multiple embedding function group. Compatible with ``func``.
        """

    @property
    @abstractmethod
    def func(self):
        """Disk caching embedding function being tested."""

    # FIXME: Test that returned embeddings could plausibly be correct.

    def test_calls_same_name_non_caching_version_if_not_cached(self):
        with self._patch_non_disk_caching_embedder() as mock:
            self.func(self.text_or_texts, data_dir=self._dir_path)

        mock.assert_called_once_with(self.text_or_texts)

    def test_saves_file_if_not_cached(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func(self.text_or_texts, data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_loads_file_if_cached(self):
        self._write_fake_data_file()

        expected_message = 'INFO:embed.cached:{name}: loaded: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self.func(self.text_or_texts, data_dir=self._dir_path)

        self.assertEqual(log_context.output, [expected_message])

    def test_saves_file_that_any_implementation_can_load(self):
        self.func(self.text_or_texts, data_dir=self._dir_path)
        message_format = 'INFO:embed.cached:{name}: loaded: {path}'

        for load_func in self.func_group:
            with self.subTest(load_func=load_func):
                expected_message = message_format.format(
                    name=load_func.__name__,
                    path=self._path,
                )

                with self.assertLogs(logger=cached.__name__) as log_context:
                    load_func(self.text_or_texts, data_dir=self._dir_path)

                self.assertEqual(log_context.output, [expected_message])

    @_audit.skip_if_unavailable
    def test_load_confirmed_by_audit_event(self):
        self._write_fake_data_file()
        expected_open_event = _audit.OpenEvent(str(self._path), 'r')

        with _audit.listening_for_open() as open_events:
            self.func(self.text_or_texts, data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    @_audit.skip_if_unavailable
    def test_save_confirmed_by_audit_event(self):
        # TODO: Decide whether to keep allowing just 'x', or if 'w' is OK too.
        expected_open_event = _audit.OpenEvent(str(self._path), 'x')

        with _audit.listening_for_open() as open_events:
            self.func(self.text_or_texts, data_dir=self._dir_path)

        self.assertIn(expected_open_event, open_events)

    def test_saved_embedding_exists(self):
        self.func(self.text_or_texts, data_dir=self._dir_path)
        self.assertTrue(self._path.is_file())

    def test_uses_default_data_dir_if_not_passed(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with patch(f'{cached.__name__}.DEFAULT_DATA_DIR', self._dir_path):
            with self.assertLogs(logger=cached.__name__) as log_context:
                self.func(self.text_or_texts)

        self.assertEqual(
            log_context.output, [expected_message],
            'DEFAULT_DATA_DIR should be used',
        )

    @property
    def _name(self):
        """Name of the disk caching embedding function being tested."""
        return self.func.__name__

    @property
    def _path(self):
        """Path of temporary test file."""
        return self._dir_path / self.filename

    def _patch_non_disk_caching_embedder(self):
        """Patch a function in ``embed`` to examine its calls."""
        embedder = getattr(embed, self._name)
        return patch(
            target=f'{embed.__name__}.{self._name}',
            wraps=embedder,
            __name__=embedder.__name__,
        )

    def _write_fake_data_file(self):
        """Create a file containing a fake embedding."""
        with open(file=self._path, mode='w', encoding='utf-8') as file:
            json.dump(obj=self.fake_data, fp=file)


class _TestDiskCachedEmbedOneBase(_TestDiskCachedCachingBase):
    """Abstract base for ``embed_one*`` group customizations."""

    @property
    def text_or_texts(self):
        return 'hola'

    @property
    def filename(self):
        return _HOLA_FILENAME

    @property
    def fake_data(self):
        """Normalized vector."""
        return [1.0] + [0.0] * (embed.DIMENSION - 1)  # Normalized vector.

    @property
    def func_group(self):
        return cached.embed_one, cached.embed_one_eu, cached.embed_one_req


class _TestDiskCachedEmbedManyBase(_TestDiskCachedCachingBase):
    """Abstract base for ``embed_many*`` group customizations."""

    @property
    def text_or_texts(self):
        return ['hola', 'hello']

    @property
    def filename(self):
        return _HOLA_HELLO_FILENAME

    @property
    def fake_data(self):
        """Two normalized vectors."""
        return [
            [1.0] + [0.0] + [0.0] * (embed.DIMENSION - 2),
            [0.0] + [1.0] + [0.0] * (embed.DIMENSION - 2),
        ]

    @property
    def func_group(self):
        return cached.embed_many, cached.embed_many_eu, cached.embed_many_req


class TestDiskCachedEmbedOne(_TestDiskCachedEmbedOneBase):
    """Tests for disk cached ``embed_one``."""

    @property
    def func(self):
        return cached.embed_one


class TestDiskCachedEmbedOneEu(_TestDiskCachedEmbedOneBase):
    """Tests for disk cached ``embed_one_eu``."""

    @property
    def func(self):
        return cached.embed_one_eu


class TestDiskCachedEmbedOneReq(_TestDiskCachedEmbedOneBase):
    """Tests for disk cached ``embed_one_req``."""

    @property
    def func(self):
        return cached.embed_one_req


class TestDiskCachedEmbedMany(_TestDiskCachedEmbedManyBase):
    """Tests for disk cached ``embed_many``."""

    @property
    def func(self):
        return cached.embed_many


class TestDiskCachedEmbedManyEu(_TestDiskCachedEmbedManyBase):
    """Tests for disk cached ``embed_many_eu``."""

    @property
    def func(self):
        return cached.embed_many_eu


class TestDiskCachedEmbedManyReq(_TestDiskCachedEmbedManyBase):
    """Tests for disk cached ``embed_many_req``."""

    @property
    def func(self):
        return cached.embed_many_req


del _TestDiskCachedCachingBase
del _TestDiskCachedEmbedOneBase
del _TestDiskCachedEmbedManyBase


if __name__ == '__main__':
    unittest.main()
