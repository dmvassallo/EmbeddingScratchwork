"""Shared base classes for embedding tests."""

from abc import ABC, abstractmethod
import pathlib
import sys
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

import embed
from tests import _helpers


class TestEmbedBase(ABC, unittest.TestCase):
    """Abstract base to provide helpers and fixtures."""

    def setUp(self):
        super().setUp()

        _helpers.maybe_cache_embeddings_in_memory.__enter__()

        self.addCleanup(
            _helpers.maybe_cache_embeddings_in_memory.__exit__,
            *sys.exc_info(),
        )

    @property
    @abstractmethod
    def func(self):
        """Embedding function being tested."""


class TestEmbedOneBase(TestEmbedBase):
    """
    Tests for ``embed.embed_one*`` functions (the non-disk-caching functions).
    """

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


class TestEmbedManyBase(TestEmbedBase):
    """
    Tests for ``embed.embed_many*`` functions (the non-disk-caching functions).
    """

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


class TestDiskCachedBase(TestEmbedBase):
    """Shared test fixture logic for all tests of disk caching versions."""

    def setUp(self):
        """Create a temporary directory."""
        super().setUp()

        # pylint: disable=consider-using-with  # tearDown cleans this up.
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._dir_path = pathlib.Path(self._temporary_directory.name)

    def tearDown(self):  # FIXME: Do this with addCleanup in setUp instead.
        """Delete the temporary directory."""
        self._temporary_directory.cleanup()
        super().tearDown()