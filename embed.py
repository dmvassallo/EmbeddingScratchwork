"""Embed function for OpenAI API experimenation."""

import os

import numpy as np
import openai

# TODO: Make this work more universally.
openai.api_key = os.environ['OPEN_AI_KEY_EMBEDDINGSCRATCHWORK']


def embed(text):
    """Embed a single piece of text."""
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)
