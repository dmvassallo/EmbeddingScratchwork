"""
API key helpers.

This enables the ``embed`` module to have an ``api_key`` property that, when
set, sets ``openai.api_key``. This is useful because code that consumes this
project's ``embed`` module shouldn't have to use ``openai`` directly or know
about ``openai.api_key``. (Setting ``openai.api_key`` to use ``requests``-based
functions, which don't use ``openai``, would be especially unintuitive.)
"""

# FIXME: Set embed.api_key eagerly rather than lazily, so its behavior doesn't
#        needlessly diverge from that of openai.api_key.

__all__ = ['KeyForwardingModule']

import os
import types

import openai


class KeyForwardingModule(types.ModuleType):
    """Module whose ``api_key`` property also sets ``openai.api_key``."""

    @property
    def api_key(self):
        """OpenAI API key."""
        try:
            return self.__api_key
        except AttributeError:
            self.set_api_key_from_environment()
            return self.__api_key

    @api_key.setter
    def api_key(self, value):
        self.__api_key = openai.api_key = value

    def set_api_key_from_environment(self):
        """
        Set ``api_key`` from the ``OPENAI_API_KEY`` environment variable.

        That happens automatically when the ``api_key`` property is first read.
        But in multithreaded scenarios, this should be called ahead of time.
        """
        self.api_key = os.getenv('OPENAI_API_KEY')  # Calls the setter.
