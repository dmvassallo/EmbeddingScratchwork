"""Shared base classes for embedding tests."""

from abc import ABC, abstractmethod
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from parameterized import parameterized

import embed
from tests import _helpers


class TestBase(unittest.TestCase):
    """Base class for all test classes in the project."""

    @classmethod
    def setUpClass(cls):
        """Make sure logging is configured as requested in the environment."""
        super().setUpClass()
        _helpers.configure_logging()

    if sys.version_info < (3, 11):
        def enterContext(self, cm):
            """
            Enter the given context manager and arrange to exit it on cleanup.

            This supplies a simplified version of ``TestCase.enterContext`` for
            versions of Python that do not have it.
            """
            context = cm.__enter__()  # pylint: disable=unnecessary-dunder-call
            self.addCleanup(lambda: cm.__exit__(*sys.exc_info()))
            return context


class TestEmbedBase(TestBase, ABC):
    """Base of all classes that test embedding functions."""

    def setUp(self):
        """
        Arrange to cache ``embed*`` calls' results in memory, if requested.

        In-memory caching is a feature of this test suite, which can be used to
        reduce the number of API calls made on CI. It should not be confused
        with on-disk caching, which is a feature of the code under test.

        In-memory caching is not done by default. It is controlled by the
        ``TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY`` environment variable. If it
        is enabled, each embedding function has its own in-memory cache, so
        bugs in some don't hide bugs in others.
        """
        super().setUp()

        if _helpers.getenv_bool('TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY'):
            self.enterContext(_helpers.cache_embeddings_in_memory)

    @property
    @abstractmethod
    def func(self):
        """Embedding function being tested."""


class TestDiskCachedBase(TestEmbedBase):
    """Shared test fixture logic for all tests of disk caching versions."""

    def setUp(self):
        """Create a temporary directory."""
        super().setUp()

        # pylint: disable-next=consider-using-with
        self.dir_path = Path(self.enterContext(TemporaryDirectory()))

    @property
    @abstractmethod
    def file_type(self):
        """File type (in file-extension form) to save and load embeddings."""


class TestEmbedOneBase(TestEmbedBase):
    """
    Tests of core ``embed.embed_one*`` functionality.

    These are the tests of the behaviors shared by all single-text embedding
    functions (regardless of whether they perform disk-based caching).
    """

    # pylint: disable=missing-function-docstring  # Tests' names describe them.

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
        ('catrun_en', 'The cat runs.'),
        ('catrun_es', 'El gato corre.'),
        ('dogwalk_en', 'The dog walks.'),
        ('dogwalk_es', 'El perro camina.'),
        ('lionsleep_en', 'The lion sleeps.'),
        ('lionsleep_es', 'El león duerme.'),
    ])
    def test_embeddings_are_normalized(self, _name, text):
        embedding = self.func(text)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(
            norm, 1.0,
            places=3,  # Allow a pretty wide margin for rounding error.
        )

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
    Tests of core ``embed_many*`` functionality.

    These are the tests of the behaviors shared by all multiple-text embedding
    functions (regardless of whether they perform disk-based caching).
    """

    def setUp(self):
        """
        Get an embeddings matrix from the ``embed_many*`` function under test.
        """
        super().setUp()

        self._many = self.func([
            'Your text string goes here',
            'The cat runs.',
            'El gato corre.',
            'The dog walks.',
            'El perro camina.',
        ])

    # pylint: disable=missing-function-docstring  # Tests' names describe them.

    def test_returns_numpy_array(self):
        with self.subTest('ndarray'):
            self.assertIsInstance(self._many, np.ndarray)
        with self.subTest('float32'):
            self.assertIsInstance(self._many[0][0], np.float32)

    def test_shape_has_model_dimension(self):
        self.assertEqual(self._many.shape, (5, embed.DIMENSION))

    def test_embeddings_are_normalized(self):
        names = ['your', 'catrun_en', 'catrun_es', 'dogwalk_en', 'dogwalk_es']
        for name, embedding in zip(names, self._many):
            with self.subTest(name):
                norm = np.linalg.norm(embedding)
                self.assertAlmostEqual(
                    norm, 1.0,
                    places=3,  # Allow a pretty wide margin for rounding error.
                )

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
