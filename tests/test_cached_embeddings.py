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


class _TestDiskCacheHitBase(_bases.TestDiskCachedBase):
    """Test fixture so embeddings are pre-cached to disk."""

    def setUp(self):
        """Copy embeddings to the temporary directory."""
        super().setUp()

        for path in pathlib.Path('tests_data').glob('*.json'):
            shutil.copy(path, self._dir_path)


class _TestDiskCacheMissBase(_bases.TestDiskCachedBase):
    pass


if __name__ == '__main__':
    unittest.main()
