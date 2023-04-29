#!/usr/bin/env python

"""
Tests of embeddings from ``embed.cached.embed*`` functions.

Those embedding functions are the versions that cache to disk. They are
otherwise like the same-named functions residing directly in ``embed``.
"""

import pathlib
import shutil
import unittest

from embed import cached
from tests import _bases


class _TestDiskCacheEmbeddingsBase(_bases.TestDiskCachedBase):
    """Base class for the embeddings tests of the disk caching versions."""

    def setUp(self):
        """Patch ``DEFAULT_DATA_DIR`` to the temporary directory."""
        super().setUp()

        self._old_data_dir = cached.DEFAULT_DATA_DIR
        cached.DEFAULT_DATA_DIR = self._dir_path

    def tearDown(self):  # FIXME: Do this with addCleanup in setUp instead.
        cached.DEFAULT_DATA_DIR = self._old_data_dir
        super().tearDown()


class _TestDiskCacheHitBase(_TestDiskCacheEmbeddingsBase):
    """Test fixture so embeddings are pre-cached to disk."""

    def setUp(self):
        """Copy embeddings to the temporary directory."""
        super().setUp()

        for path in pathlib.Path('tests_data').glob('*.json'):
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


class _TestDiskCacheMissBase(_TestDiskCacheEmbeddingsBase):
    pass


if __name__ == '__main__':
    unittest.main()
