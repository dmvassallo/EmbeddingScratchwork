"""Embed function for OpenAI API experimentation."""

import os

import numpy as np
import openai


def embed_one(text):
    """Embed a single piece of text."""
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)
