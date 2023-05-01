#!/usr/bin/env python

"""
Tests of embeddings from ``embed.cached.embed*`` functions.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

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

        for path in Path('tests_data').glob('*.json'):
            shutil.copy(path, self._dir_path)


class TestDiskCacheHitEmbedOne(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one


class TestDiskCacheHitEmbedOneEu(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one_eu`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one_eu


class TestDiskCacheHitEmbedOneReq(
    _bases.TestEmbedOneBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_one_req`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_one_req


class TestDiskCacheHitEmbedMany(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many


class TestDiskCacheHitEmbedManyEu(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many_eu`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many_eu


class TestDiskCacheHitEmbedManyReq(
    _bases.TestEmbedManyBase,
    _TestDiskCacheHitBase,
):
    """Tests for disk cached ``embed_many_req`` with embeddings pre-cached."""

    @property
    def func(self):
        return cached.embed_many_req


class TestDiskCacheMissEmbedOne(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_one``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_one


class TestDiskCacheMissEmbedOneEu(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_one_eu``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_one_eu


class TestDiskCacheMissEmbedOneReq(
    _bases.TestEmbedOneBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_one_req``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_one_req


class TestDiskCacheMissEmbedMany(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_many``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_many


class TestDiskCacheMissEmbedManyEu(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_many_eu``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_many_eu


class TestDiskCacheMissEmbedManyReq(
    _bases.TestEmbedManyBase,
    _TestDiskCacheEmbeddingsBase,
):
    """Tests for disk cached ``embed_many_req``, embeddings not pre-cached."""

    @property
    def func(self):
        return cached.embed_many_req


if __name__ == '__main__':
    unittest.main()
