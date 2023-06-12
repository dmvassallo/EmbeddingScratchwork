#!/usr/bin/env python

"""Tests for the ``api_key`` property of the ``embed`` module."""

import contextlib
import os
from pathlib import Path
import string
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import openai
from parameterized import parameterized

import embed
from embed._keys import _get_key_if_available
from tests import _bases


class TestApiKey(_bases.TestBase):
    """Tests for ``embed.api_key``."""

    def setUp(self):
        """Save api_key attributes. Also pre-patch them, for log redaction."""
        super().setUp()

        # This cannot be done straightforwardly with unittest.mock.patch
        # because that expects to be able to delete attributes, and the
        # embed.api_key property (deliberately) has no deleter.
        self._real_key_openai = openai.api_key
        self._real_key_embed = embed.api_key
        openai.api_key = 'sk-fake_redact_outer'
        embed.api_key = 'sk-fake_redact_inner'

    def tearDown(self):
        """Unpatch api_key attributes."""
        embed.api_key = self._real_key_embed
        openai.api_Key = self._real_key_openai

        super().tearDown()

    @parameterized.expand([
        ('str', 'sk-fake_setting_sets'),
        ('none', None),
    ])
    def test_setting_on_embed_sets_on_openai(self, _name, pretend_key):
        """Setting ``embed.api_key`` sets both it and ``openai.api_key``."""
        embed.api_key = pretend_key
        with self.subTest('embed.api_key'):
            self.assertEqual(embed.api_key, pretend_key)
        with self.subTest('openai.api_key'):
            self.assertEqual(openai.api_key, pretend_key)

    @parameterized.expand([
        ('str', 'sk-fake_setting_does_not_set'),
        ('none', None),
    ])
    def test_setting_on_openai_does_not_set_on_embed(self, _name, pretend_key):
        """Setting ``open.api_key`` does not change ``embed.api_key``."""
        openai.api_key = pretend_key
        self.assertNotEqual(embed.api_key, pretend_key)


_parameterize_by_distance = parameterized.expand([
    (str(distance), distance) for distance in (1, 2, 5, 10)
])
"""Parameterize a test case by a number of nested subdirectories."""


class TestGetKeyIfAvailable(_bases.TestBase):
    """
    Tests for the non-public ``embed._keys._get_key_if_available`` function.

    These tests test the code that is used to determine the automatic initial
    value of ``embed.api_key``.
    """

    def setUp(self):
        super().setUp()

        self.enterContext(contextlib.chdir(
            self.enterContext(TemporaryDirectory()),
        ))

        self.enterContext(
            patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-fake_from_env'}),
        )

    def test_uses_env_var_when_no_key_file(self):
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_env')

    def test_uses_env_var_instead_of_key_file(self):
        Path('.api_key').write_text('sk-fake_from_file', encoding='utf-8')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_env')

    def test_uses_key_file_in_cwd_when_no_env_var(self):
        del os.environ['OPENAI_API_KEY']
        Path('.api_key').write_text('sk-fake_from_file', encoding='utf-8')
        result = _get_key_if_available()
        self.assertEqual(result, 'sk-fake_from_file')

    def test_none_found_when_no_env_var_nor_key_file(self):
        del os.environ['OPENAI_API_KEY']
        result = _get_key_if_available()
        self.assertIsNone(result)

    def test_key_file_in_parent_when_no_repo_not_used(self):
        del os.environ['OPENAI_API_KEY']
        Path('.api_key').write_text('sk-fake_from_file', encoding='utf-8')

        subdir = Path('subdir')
        subdir.mkdir()
        os.chdir(subdir)

        result = _get_key_if_available()
        self.assertIsNone(result)

    @_parameterize_by_distance
    def test_key_file_in_ancestor_outside_repo_not_used(self, _name, distance):
        del os.environ['OPENAI_API_KEY']
        Path('.api_key').write_text('sk-fake_from_file', encoding='utf-8')

        for one_letter_dir_name in string.ascii_lowercase[:distance]:
            subdir = Path(one_letter_dir_name)
            subdir.mkdir()
            os.chdir(subdir)

        result = _get_key_if_available()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
