#!/usr/bin/env python

"""
Tests for behavior that all embedding functions should have.

This consists mostly of tests of embeddings from ``embed.embed*`` functions.
This does not test the disk caching versions; see ``test_cached_embeddings``.
"""

import unittest

from parameterized import parameterized

import embed
from tests import _bases


class TestStats(_bases.TestBase):
    """Tests for model dimension and token encoding facilites in ``embed``."""

    def test_model_dimension_is_1536(self):
        """``DIMENSION``'s value is correct for ``text-embedding-ada-002``."""
        self.assertEqual(embed.DIMENSION, 1536)

    @parameterized.expand([
        ('The cat runs.', 4),
        ('El gato corre.', 5),
        ('The dog walks.', 4),
        ('El perro camina.', 6),
        ('Ms. Jones visited the graveyard where she had first explained '
         'nautical twilight to her curiously intelligent capybara.', 23),
        ('aerodynamic', 3),
        (' aerodynamic', 2),  # Often fewer tokens with a leading space.
        ('孫子兵法', 6),  # With some languages, more tokens than characters.
    ])
    def test_count_tokens_counts_cl100k_base_tokens(self, text, expected):
        """``count_tokens`` counts in the ``cl100k_base`` encoding."""
        actual = embed.count_tokens(text)
        self.assertEqual(actual, expected)


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
