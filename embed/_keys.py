"""
API key helpers.

This enables the ``embed`` module to have an ``api_key`` property that, when
set, sets ``openai.api_key``. This is useful because code that consumes this
project's ``embed`` module shouldn't have to use ``openai`` directly or know
about ``openai.api_key``. (Setting ``openai.api_key`` to use ``requests``-based
functions, which don't use ``openai``, would be especially unintuitive.)

Code within the ``embed`` module itself may access ``_keys.api_key``.
"""

__all__ = ['api_key', 'initialize']

import os
import sys
import types

import openai

api_key = None
"""OpenAI API key. This should only be accessed from ``__init__.py``."""


def initialize(module_or_name):
    """
    Give the module an ``api_key`` property and set it from the environment.

    Setting the property sets ``openai.api_key`` (including this first time).
    """
    if isinstance(module_or_name, str):  # Because no match-case before 3.10.
        module = sys.modules[module_or_name]
    elif isinstance(module_or_name, types.ModuleType):
        module = module_or_name
    else:
        raise TypeError(f'module_or_name is {type(module_or_name).__name__!r}')

    # Give the module an api_key property that updates openai.api_key when set.
    module.__class__ = _ModuleWithApiKeyProperty

    # Set the property from the environment.
    module.api_key = os.getenv('OPENAI_API_KEY')


class _ModuleWithApiKeyProperty(types.ModuleType):
    """A module whose ``api_key`` property also sets ``openai.api_key``."""

    @property
    def api_key(self):
        """OpenAI API key."""
        return api_key

    @api_key.setter
    def api_key(self, value):
        global api_key
        api_key = openai.api_key = value
