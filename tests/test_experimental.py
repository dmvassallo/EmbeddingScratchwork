#!/usr/bin/env python

"""
Tests for embedding functions.

This is a work in progress using inheritance instead of parameterized_class.
"""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

from abc import ABC, abstractmethod
import unittest

import numpy as np
from parameterized import parameterized

import embed
from tests import _helpers

_helpers.configure_logging()


class TestConstants(unittest.TestCase):
    """Tests for public constants in ``embed``."""

    def test_model_dimension_is_1536(self):
        self.assertEqual(embed.DIMENSION, 1536)


class _TestEmbedOneBase(ABC, unittest.TestCase):
    """Tests for ``embed_one``, ``embed_one_eu``, and ``embed_one_req``."""

    def test_returns_numpy_array(self):
        result = self.func('Your text string goes here')
        with self.subTest('ndarray'):
            self.assertIsInstance(result, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(result[0], np.float32)

    def test_shape_is_model_dimension(self):
        result = self.func('Your text string goes here')
        self.assertEqual(result.shape, (embed.DIMENSION,))

    @parameterized.expand([
        ('catrun', 'The cat runs.', 'El gato corre.'),
        ('dogwalk', 'The dog walks.', 'El perro camina.'),
        ('lionsleep', 'The lion sleeps.', 'El león duerme.'),
    ])
    def test_en_and_es_sentence_are_very_similar(
            self, _name, text_en, text_es):
        embedding_en = self.func(text_en)
        embedding_es = self.func(text_es)
        result = np.dot(embedding_en, embedding_es)
        self.assertGreaterEqual(result, 0.9)

    def test_different_meanings_are_dissimilar(self):
        sentence_one = self.func('Your text string goes here')
        sentence_two = self.func('The cat runs.')
        result = np.dot(sentence_one, sentence_two)
        self.assertLess(result, 0.8)

    @property
    @abstractmethod
    def func(self):
        """Embedding function being tested."""


class TestEmbedOne(_TestEmbedOneBase):
    """Tests for ``embed_one``."""

    @property
    def func(self):
        return embed.embed_one


class TestEmbedOneEu(_TestEmbedOneBase):
    """Tests for ``embed_one_eu``."""

    @property
    def func(self):
        return embed.embed_one_eu


class TestEmbedOneReq(_TestEmbedOneBase):
    """Tests for ``embed_one_req``."""

    @property
    def func(self):
        return embed.embed_one_req


del _TestEmbedOneBase


if __name__ == '__main__':
    unittest.main()
