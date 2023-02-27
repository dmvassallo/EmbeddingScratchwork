"""Tests for the embed module."""

# FIXME: Reduce calls to API.

import unittest

import numpy as np
from parameterized import parameterized

from embed import embed_one, embed_many


class TestEmbedOne(unittest.TestCase):
    """Tests for embed_one."""

    def test_returns_numpy_array(self):
        result = embed_one("Your text string goes here")
        with self.subTest('ndarray'):
            self.assertIsInstance(result, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(result[0], np.float32)

    def test_shape_is_model_dimension(self):
        result = embed_one("Your text string goes here")
        self.assertEqual(result.shape, (1536,))

    @parameterized.expand([
        ("catrun", "The cat runs.", "El gato corre."),
        ("dogwalk", "The dog walks.", "El perro camina."),
        ("lionsleep", "The lion sleeps.", "El león duerme."),
    ])
    def test_en_and_es_sentence_are_very_similar(
            self, _name, text_en, text_es):
        embedding_en = embed_one(text_en)
        embedding_es = embed_one(text_es)
        result = np.dot(embedding_en, embedding_es)
        self.assertGreaterEqual(result, 0.9)

    def test_different_meanings_are_dissimilar(self):
        sentence_one = embed_one("Your text string goes here")
        sentence_two = embed_one("The cat runs.")
        result = np.dot(sentence_one, sentence_two)
        self.assertLess(result, 0.8)
