"""Embed function for OpenAI API experimentation."""

import numpy as np
import openai


def embed_one(text):
    """Embed a single piece of text."""
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)


def embed_many(texts):
    """Embed multiple pieces of text."""
    response = openai.Embedding.create(
        input=texts,
        model='text-embedding-ada-002',
    )
    # FIXME: Deal with the out of order case.
    embeds = [d['embedding'] for d in response['data']]
    return np.array(embeds, dtype=np.float32)
