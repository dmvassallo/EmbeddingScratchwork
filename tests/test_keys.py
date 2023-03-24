#!/usr/bin/env python

"""Tests for the ``api_key`` property of the ``embed`` module."""

import unittest

import openai
from parameterized import parameterized

import embed


class TestApiKey(unittest.TestCase):
    """Tests for ``embed.api_key``."""

    def setUp(self):
        """Save api_key attributes. Also pre-patch them, for log redaction."""
        # This cannot be done straightforwardly with unittest.mock.patch
        # because that expects to be able to delete attributes, and the
        # embed.api_key property (deliberately) has no deleter.
        self._real_key_openai = openai.api_key
        self._real_key_embed = embed.api_key
        openai.api_key = 'sk-fake-redact-outer'
        embed.api_key = 'sk-fake-redact-inner'

    def tearDown(self):
        """Unpatch api_key attributes."""
        embed.api_key = self._real_key_embed
        openai.api_Key = self._real_key_openai

    @parameterized.expand([
        ('str', 'sk-fake-setting-sets'),
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
        ('str', 'sk-fake-setting-does-not-set'),
        ('none', None),
    ])
    def test_setting_on_openai_does_not_set_on_embed(self, name, pretend_key):
        """Setting ``open.api_key`` does not change ``embed.api_key``."""
        openai.api_key = pretend_key
        self.assertNotEqual(embed.api_key, pretend_key)


if __name__ == '__main__':
    unittest.main()
