#!/usr/bin/env python

"""
Tests of disk caching behavior of ``embed.cached.embed*`` functions.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

from abc import abstractmethod
import json
import unittest
from unittest.mock import ANY, Mock, patch

import numpy as np
import safetensors.numpy
import subaudit

import embed
from embed import cached
from tests import _bases

_HOLA_BASENAME = (
    'b58e4a60c963f8b3c43d83cc9245020ce71d8311fa2f48cfd36deed6f472a71b'
)
"""Filename stem that would be generated from the input ``'hola'``."""

_HOLA_HELLO_BASENAME = (
    '2e41e52e67421c1d106bb8a5b9225ad1143761240862ed61e5be5ed20f39f2fd'
)
"""
Filename stem that would be generated from the input ``['hola', 'hello']``.
"""


class _TestDiskCachedCachingBase(_bases.TestDiskCachedBase):
    """
    Tests specific to caching behavior of ``embed.cached.embed*`` functions.
    """

    @property
    @abstractmethod
    def text_or_texts(self):
        """Input to the embedding functions."""

    @property
    @abstractmethod
    def basename(self):
        """
        Filename stem that should be generated from input ``text_or_texts``.
        """

    @property
    @abstractmethod
    def fake_data(self):
        """Fake data for testing loads from a file."""

    @abstractmethod
    def write_fake_data_file(self):
        """Write a fake data file. Implementations should use ``fake_data``."""

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

    # pylint: disable=missing-function-docstring  # Tests' names describe them.

    def test_calls_same_name_non_caching_version_if_not_cached(self):
        with self._patch_non_disk_caching_embedder() as mock:
            self._call_caching_embedder()

        mock.assert_called_once_with(self.text_or_texts)

    def test_saves_file_if_not_cached(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self._call_caching_embedder()

        self.assertEqual(log_context.output, [expected_message])

    def test_loads_file_if_cached(self):
        self.write_fake_data_file()

        expected_message = 'INFO:embed.cached:{name}: loaded: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with self.assertLogs(logger=cached.__name__) as log_context:
            self._call_caching_embedder()

        self.assertEqual(log_context.output, [expected_message])

    def test_saves_file_that_any_implementation_can_load(self):
        self._call_caching_embedder()
        message_format = 'INFO:embed.cached:{name}: loaded: {path}'

        for load_func in self.func_group:
            with self.subTest(load_func=load_func):
                expected_message = message_format.format(
                    name=load_func.__name__,
                    path=self._path,
                )

                with self.assertLogs(logger=cached.__name__) as log_context:
                    self._call_caching_embedder(func=load_func)

                self.assertEqual(log_context.output, [expected_message])

    def test_load_confirmed_by_audit_event(self):
        self.write_fake_data_file()

        with subaudit.listening('open', Mock()) as listener:
            self._call_caching_embedder()

        listener.assert_any_call(str(self._path), 'r', ANY)

    def test_save_confirmed_by_audit_event(self):
        with subaudit.listening('open', Mock()) as listener:
            self._call_caching_embedder()

        listener.assert_any_call(str(self._path), 'w', ANY)

    def test_saved_embedding_exists(self):
        self._call_caching_embedder()
        self.assertTrue(self._path.is_file())

    def test_uses_default_data_dir_if_not_passed(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with patch(f'{cached.__name__}.DEFAULT_DATA_DIR', self.dir_path):
            with self.assertLogs(logger=cached.__name__) as log_context:
                self.func(self.text_or_texts, file_type=self.file_type)

        self.assertEqual(
            log_context.output, [expected_message],
            'DEFAULT_DATA_DIR should be used',
        )

    def test_uses_default_file_type_if_not_passed(self):
        expected_message = 'INFO:embed.cached:{name}: saved: {path}'.format(
            name=self._name,
            path=self._path,
        )

        with patch(f'{cached.__name__}.DEFAULT_FILE_TYPE', self.file_type):
            with self.assertLogs(logger=cached.__name__) as log_context:
                self.func(self.text_or_texts, data_dir=self.dir_path)

        self.assertEqual(
            log_context.output, [expected_message],
            'DEFAULT_FILE_TYPE should be used',
        )

    @property
    def _name(self):
        """Name of the disk caching embedding function being tested."""
        return self.func.__name__

    @property
    def _path(self):
        """Path of temporary test file."""
        return self.dir_path / f'{self.basename}.{self.file_type}'

    def _call_caching_embedder(self, *, func=None):
        """Call a caching embedder. Pass usual per-subclass test arguments."""
        if func is None:  # We usually call self.func. Use that as a default.
            func = self.func

        return func(
            self.text_or_texts,
            data_dir=self.dir_path,
            file_type=self.file_type,
        )

    def _patch_non_disk_caching_embedder(self):
        """Patch a function in ``embed`` to examine its calls."""
        embedder = getattr(embed, self._name)
        return patch(
            target=f'{embed.__name__}.{self._name}',
            wraps=embedder,
            __name__=embedder.__name__,
        )


class _TestDiskCachedEmbedOneBase(_TestDiskCachedCachingBase):
    """Abstract base for ``embed_one*`` group customizations."""

    @property
    def text_or_texts(self):
        return 'hola'

    @property
    def basename(self):
        return _HOLA_BASENAME

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
    def basename(self):
        return _HOLA_HELLO_BASENAME

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


class _TestDiskCachedJsonBase(_TestDiskCachedCachingBase):
    """Abstract base for tests using JSON serialization."""

    @property
    def file_type(self):
        return 'json'

    def write_fake_data_file(self):
        """Create a JSON file containing a fake embedding."""
        with open(file=self._path, mode='w', encoding='utf-8') as file:
            json.dump(obj=self.fake_data, fp=file)


class _TestDiskCachedSafetensorsBase(_TestDiskCachedCachingBase):
    """Abstract base for tests using safetensors serialization."""

    @property
    def file_type(self):
        return 'safetensors'

    def write_fake_data_file(self):
        """Write a safetensors file containing a fake embedding."""
        data = np.array(self.fake_data, dtype=np.float32)
        safetensors.numpy.save_file({'embeddings': data}, self._path)

    @unittest.expectedFailure  # TODO: Look into observing native file access.
    def test_load_confirmed_by_audit_event(self):
        super().test_load_confirmed_by_audit_event()

    @unittest.expectedFailure  # TODO: Look into observing native file access.
    def test_save_confirmed_by_audit_event(self):
        super().test_save_confirmed_by_audit_event()


class TestDiskCachedEmbedOneJson(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_one`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_one


class TestDiskCachedEmbedOneSafetensors(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedSafetensorsBase,
):
    """Tests for disk cached ``embed_one`` with safetensors serialization."""

    @property
    def func(self):
        return cached.embed_one


class TestDiskCachedEmbedOneEuJson(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_one_eu`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_one_eu


class TestDiskCachedEmbedOneEuSafetensors(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedSafetensorsBase,
):
    """
    Tests for disk cached ``embed_one_eu`` with safetensors serialization.
    """

    @property
    def func(self):
        return cached.embed_one_eu


class TestDiskCachedEmbedOneReqJson(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_one_req`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_one_req


class TestDiskCachedEmbedOneReqSafetensors(
    _TestDiskCachedEmbedOneBase,
    _TestDiskCachedSafetensorsBase,
):
    """
    Tests for disk cached ``embed_one_req`` with safetensors serialization.
    """

    @property
    def func(self):
        return cached.embed_one_req


class TestDiskCachedEmbedManyJson(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_many`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_many


class TestDiskCachedEmbedManySafetensors(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedSafetensorsBase,
):
    """Tests for disk cached ``embed_many`` with safetensors serialization."""

    @property
    def func(self):
        return cached.embed_many


class TestDiskCachedEmbedManyEuJson(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_many_eu`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_many_eu


class TestDiskCachedEmbedManyEuSafetensors(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedSafetensorsBase,
):
    """
    Tests for disk cached ``embed_many_eu`` with safetensors serialization.
    """

    @property
    def func(self):
        return cached.embed_many_eu


class TestDiskCachedEmbedManyReqJson(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedJsonBase,
):
    """Tests for disk cached ``embed_many_req`` with JSON serialization."""

    @property
    def func(self):
        return cached.embed_many_req


class TestDiskCachedEmbedManyReqSafetensors(
    _TestDiskCachedEmbedManyBase,
    _TestDiskCachedSafetensorsBase,
):
    """
    Tests for disk cached ``embed_many_req`` with safetensors serialization.
    """

    @property
    def func(self):
        return cached.embed_many_req


del _TestDiskCachedCachingBase
del _TestDiskCachedEmbedOneBase
del _TestDiskCachedEmbedManyBase
del _TestDiskCachedJsonBase
del _TestDiskCachedSafetensorsBase


if __name__ == '__main__':
    unittest.main()
