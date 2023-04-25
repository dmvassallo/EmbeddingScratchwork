#!/usr/bin/env python

"""Tests for behavior that all embedding functions should have."""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

import unittest

import embed
from tests import _bases, _helpers

_helpers.configure_logging()


class TestConstants(unittest.TestCase):
    """Tests for public constants in ``embed``."""

    def test_model_dimension_is_1536(self):
        self.assertEqual(embed.DIMENSION, 1536)


class TestEmbedOne(_bases.TestEmbedOneBase):
    """Tests for ``embed_one``."""

    @property
    def func(self):
        return embed.embed_one


class TestEmbedOneEu(_bases.TestEmbedOneBase):
    """Tests for ``embed_one_eu``."""

    @property
    def func(self):
        return embed.embed_one_eu


class TestEmbedOneReq(_bases.TestEmbedOneBase):
    """Tests for ``embed_one_req``."""

    @property
    def func(self):
        return embed.embed_one_req


class TestEmbedMany(_bases.TestEmbedManyBase):
    """Tests for ``embed_many``."""

    @property
    def func(self):
        return embed.embed_many


class TestEmbedManyEu(_bases.TestEmbedManyBase):
    """Tests for ``embed_many_eu``."""

    @property
    def func(self):
        return embed.embed_many_eu


class TestEmbedManyReq(_bases.TestEmbedManyBase):
    """Tests for ``embed_many_req``."""

    @property
    def func(self):
        return embed.embed_many_req


if __name__ == '__main__':
    unittest.main()
