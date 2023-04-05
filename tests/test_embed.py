#!/usr/bin/env python

"""
Tests for embedding functions residing directly in the ``embed`` module.

Those embedding functions are the ones that do not cache to disk.
"""

# pylint: disable=missing-function-docstring
# All test methods have self-documenting names.

from typing import Any
import unittest

import numpy as np
from parameterized import parameterized

import embed
from tests import _helpers

_helpers.configure_logging()


@_helpers.parameterize_from_suppliers(
    lambda: embed.embed_one,
    lambda: embed.embed_one_eu,
    lambda: embed.embed_one_req,
)
@_helpers.maybe_cache_embeddings_in_memory
class TestEmbedOne(unittest.TestCase):
    """Tests for ``embed_one``, ``embed_one_eu``, and ``embed_one_req``."""

    func: Any

    def test_returns_numpy_array(self):
        result = self.func('Your text string goes here')
        with self.subTest('ndarray'):
            self.assertIsInstance(result, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(result[0], np.float32)

    def test_shape_is_model_dimension(self):
        result = self.func('Your text string goes here')
        self.assertEqual(result.shape, (1536,))

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


@_helpers.parameterize_from_suppliers(
    lambda: embed.embed_many,
    lambda: embed.embed_many_eu,
    lambda: embed.embed_many_req,
)
class TestEmbedMany(unittest.TestCase):
    """Tests for ``embed_many``, ``embed_many_eu``, and ``embed_many_req``."""

    func: Any

    @_helpers.maybe_cache_embeddings_in_memory
    def setUp(self):
        self._many = self.func([
            'Your text string goes here',
            'The cat runs.',
            'El gato corre.',
            'The dog walks.',
            'El perro camina.',
        ])

    def test_returns_numpy_array(self):
        with self.subTest('ndarray'):
            self.assertIsInstance(self._many, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(self._many[0][0], np.float32)

    def test_shape_is_model_dimension(self):
        self.assertEqual(self._many.shape, (5, 1536))

    def test_en_and_es_sentences_are_very_similar(self):
        with self.subTest('catrun'):
            result = np.dot(self._many[1], self._many[2])
            self.assertGreaterEqual(result, 0.9)
        with self.subTest('dogwalk'):
            result = np.dot(self._many[3], self._many[4])
            self.assertGreaterEqual(result, 0.9)

    def test_different_meanings_are_dissimilar(self):
        result = np.dot(self._many[0], self._many[1])
        self.assertLess(result, 0.8)


if __name__ == '__main__':
    unittest.main()
