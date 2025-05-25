"""
Utility functions for embedding operations.

This module provides utility functions for working with embeddings,
including conversion between different formats and vector operations.
"""

from __future__ import annotations

import functools
import logging
from typing import List

import numpy as np
from litellm.exceptions import ContextWindowExceededError

from ..protocols.base import EmbeddingFunction


def truncate_if_context_exceeded(embedding_fn: EmbeddingFunction) -> EmbeddingFunction:
    """
    Decorator for embedding functions that adds error handling for context window exceeded errors.

    This decorator wraps an embedding function to:
    1. Validate that the input text is a string
    2. Handle ContextWindowExceededError by retrying with half the text
    3. Extract the embedding from the response structure

    Args:
        embedding_fn: The embedding function to wrap

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(embedding_fn)
    def wrapper(model: str, text: str) -> List[float]:
        assert isinstance(text, str), "Text must be a string"

        try:
            return embedding_fn(model, text)
        except ContextWindowExceededError:
            logging.info("Context window exceeded, retrying with half the text")
            return wrapper(model, text[int(len(text) / 2) :])

    return wrapper


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between 0 and 1
    """

    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def embedding_to_bytes(embedding: List[float]) -> bytes:
    """
    Convert a list of floats to a bytes representation.

    Args:
        embedding: List of float values representing an embedding

    Returns:
        Bytes representation of the embedding
    """
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array.tobytes()


def bytes_to_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Convert bytes representation back to a list of floats.

    Args:
        embedding_bytes: Bytes representation of an embedding

    Returns:
        List of float values representing the embedding
    """
    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
    return embedding_array.tolist()
