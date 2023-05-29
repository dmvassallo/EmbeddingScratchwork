#!/usr/bin/env python

"""
Tests of embeddings from ``embed.cached.embed*`` functions.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

from pathlib import Path
import shutil
import unittest
import unittest.mock

from embed import cached
from tests import _bases


class _TestDiskCacheEmbeddingsBase(_bases.TestDiskCachedBase):
    """Base class for the embeddings tests of the disk caching versions."""

    def setUp(self):
        """Patch ``DEFAULT_DATA_DIR`` and ``DEFAULT_FILE_TYPE``."""
        super().setUp()
        self._patch('DEFAULT_DATA_DIR', self._dir_path)
        self._patch('DEFAULT_FILE_TYPE', self.file_type)

    def _patch(self, attribute_name, new_value):
        """Monkey-patch an ``embed.cached`` attribute. Unpatch on cleanup."""
        target = f'{cached.__name__}.{attribute_name}'
        self.enterContext(unittest.mock.patch(target, new_value))


class _TestDiskCacheHitBase(_TestDiskCacheEmbeddingsBase):
    """Test fixture so embeddings are pre-cached to disk."""

    def setUp(self):
        """Copy embeddings to the temporary directory."""
        super().setUp()

        for file_type in 'json', 'safetensors':
            for path in Path('tests_data').glob(f'*.{file_type}'):
                shutil.copy(path, self._dir_path)


class TestDiskCacheHitEmbedOneJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedOneSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_one``, safetensors embeddings pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheHitEmbedOneEuJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one_eu``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one_eu

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedOneEuSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_one_eu``, safetensors embeddings pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_eu

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheHitEmbedOneReqJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one_req``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one_req

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedOneReqSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_one_req``, safetensors embeddings pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_req

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheHitEmbedManyJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedManySafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_many``, safetensors embeddings pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheHitEmbedManyEuJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many_eu``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many_eu

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedManyEuSafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_many_eu``, safetensors embeddings pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_eu

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheHitEmbedManyReqJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many_req``, JSON embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many_req

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheHitEmbedManyReqSafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """
    Tests for disk cached ``embed_many_req``, safetensors embeddings
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_req

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedOneJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_one``, JSON embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_one

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedOneSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_one``, safetensors embeddings not pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedOneEuJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_one_eu``, JSON embeddings not pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_eu

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedOneEuSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_one_eu``, safetensors embeddings not
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_eu

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedOneReqJson(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_one_req``, JSON embeddings not pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_req

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedOneReqSafetensors(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_one_req``, safetensors embeddings not
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_one_req

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedManyJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_many``, JSON embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_many

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedManySafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_many``, safetensors embeddings not
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedManyEuJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_many_eu``, JSON embeddings not pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_eu

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedManyEuSafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_many_eu``, safetensors embeddings not
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_eu

    @property
    def file_type(self):
        return 'safetensors'


class TestDiskCacheMissEmbedManyReqJson(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_many_req``, JSON embeddings not pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_req

    @property
    def file_type(self):
        return 'json'


class TestDiskCacheMissEmbedManyReqSafetensors(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """
    Tests for disk cached ``embed_many_req``, safetensors embeddings not
    pre-cached.
    """

    @property
    def func(self):
        return cached.embed_many_req

    @property
    def file_type(self):
        return 'safetensors'


if __name__ == '__main__':
    unittest.main()
