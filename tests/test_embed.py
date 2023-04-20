#!/usr/bin/env python

"""Tests for behavior that all embedding functions should have."""

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


class _TestEmbedBase(ABC, unittest.TestCase):
    """Abstract base to provide helpers and fixtures."""

    def setUp(self):
        _helpers.maybe_cache_embeddings_in_memory.__enter__()

        self.addCleanup(
            _helpers.maybe_cache_embeddings_in_memory.__exit__,
            None, None, None,
        )

    @property
    @abstractmethod
    def func(self):
        """Embedding function being tested."""


class _TestEmbedOneBase(_TestEmbedBase):
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


class _TestEmbedManyBase(_TestEmbedBase):
    """Tests for ``embed_many``, ``embed_many_eu``, and ``embed_many_req``."""

    def setUp(self):
        super().setUp()

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

    def test_shape_has_model_dimension(self):
        self.assertEqual(self._many.shape, (5, embed.DIMENSION))

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


class TestEmbedMany(_TestEmbedManyBase):
    """Tests for ``embed_many``."""

    @property
    def func(self):
        return embed.embed_many


class TestEmbedManyEu(_TestEmbedManyBase):
    """Tests for ``embed_many_eu``."""

    @property
    def func(self):
        return embed.embed_many_eu


class TestEmbedManyReq(_TestEmbedManyBase):
    """Tests for ``embed_many_req``."""

    @property
    def func(self):
        return embed.embed_many_req


del _TestEmbedOneBase, _TestEmbedManyBase


if __name__ == '__main__':
    unittest.main()
