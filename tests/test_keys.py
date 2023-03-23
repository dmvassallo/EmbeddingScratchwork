#!/usr/bin/env python

"""Tests for the ``api_key`` property of the ``embed`` module."""

import unittest
import unittest.mock

import openai

import embed


class TestApiKey(unittest.TestCase):
    """Tests for ``embed.api_key``."""

    @unittest.mock.patch('openai.api_key', 'sk-fake-redact-outer')
    @unittest.mock.patch('embed.api_key', 'sk-fake-redact-inner')
    def test_setting_sets_openai_api_key(self):
        """Setting ``embed.api_key`` sets both it and ``openai.api_key``."""
        pretend_key = 'sk-fake-setting-sets'
        embed.api_key = pretend_key
        with self.subTest('embed.api_key'):
            self.assertEqual(embed.api_key, pretend_key)
        with self.subTest('openai.api_key'):
            self.assertEqual(openai.api_key, pretend_key)


if __name__ == '__main__':
    unittest.main()
