"""Tests for the embed module."""

# FIXME: Reduce calls to API.

import unittest

import numpy as np

from embed import embed_one, embed_many


class TestEmbedOne(unittest.TestCase):

    def test_returns_numpy_array(self):
        result = embed_one("Your text string goes here")
        with self.subTest('ndarray'):
            self.assertIsInstance(result, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(result[0], np.float32)

    def test_shape_is_model_dimension(self):
        result = embed_one("Your text string goes here")
        self.assertEqual(result.shape, (1536,))

    def test_en_and_es_sentence_are_very_similar(self):
        catrun_en = embed_one("The cat runs.")
        catrun_es = embed_one("El gato corre.")
        result = np.dot(catrun_en, catrun_es)
        self.assertGreaterEqual(result, 0.9)
