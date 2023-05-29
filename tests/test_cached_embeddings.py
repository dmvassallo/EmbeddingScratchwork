#!/usr/bin/env python

"""
Tests of embeddings from ``embed.cached.embed*`` functions.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

# FIXME: The safetensors tests wrongly test with JSON.

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from embed import cached
from tests import _bases


class _TestDiskCacheEmbeddingsBase(_bases.TestDiskCachedBase):
    """Base class for the embeddings tests of the disk caching versions."""

    def setUp(self):
        """Patch ``DEFAULT_DATA_DIR`` to the temporary directory."""
        super().setUp()

        self.enterContext(
            patch(f'{cached.__name__}.DEFAULT_DATA_DIR', self._dir_path),
        )


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
