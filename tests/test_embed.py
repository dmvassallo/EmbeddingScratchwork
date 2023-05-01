#!/usr/bin/env python

"""
Tests for behavior that all embedding functions should have.

This consists mostly of tests of embeddings from ``embed.embed*`` functions.
This does not test the disk caching versions; see ``test_cached_embeddings``.
"""

import unittest

import embed
from tests import _bases


class TestConstants(_bases.TestBase):
    """Tests for public constants in ``embed``."""

    def test_model_dimension_is_1536(self):
        """``DIMENSION``'s value is correct for ``text-embedding-ada-002``."""
        self.assertEqual(embed.DIMENSION, 1536)


class TestEmbedOne(_bases.TestEmbedOneBase):
    """Tests for the non-disk-caching ``embed_one``."""

    @property
    def func(self):
        return embed.embed_one


class TestEmbedOneEu(_bases.TestEmbedOneBase):
    """Tests for the non-disk-caching ``embed_one_eu``."""

    @property
    def func(self):
        return embed.embed_one_eu


class TestEmbedOneReq(_bases.TestEmbedOneBase):
    """Tests for the non-disk-caching ``embed_one_req``."""

    @property
    def func(self):
        return embed.embed_one_req


class TestEmbedMany(_bases.TestEmbedManyBase):
    """Tests for the non-disk-caching ``embed_many``."""

    @property
    def func(self):
        return embed.embed_many


class TestEmbedManyEu(_bases.TestEmbedManyBase):
    """Tests for the non-disk-caching ``embed_many_eu``."""

    @property
    def func(self):
        return embed.embed_many_eu


class TestEmbedManyReq(_bases.TestEmbedManyBase):
    """Tests for the non-disk-caching ``embed_many_req``."""

    @property
    def func(self):
        return embed.embed_many_req


if __name__ == '__main__':
    unittest.main()
