"""
Utility functions for embedding operations.

This module provides utility functions for working with embeddings,
including conversion between different formats and vector operations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

import numpy as np
from litellm import ContextWindowExceededError
from litellm import embedding as litellm_embedding


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

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


def embedding_to_string(embedding: List[float]) -> str:
    """
    Convert an embedding to a comma-separated string.

    Args:
        embedding: List of float values representing an embedding

    Returns:
        Comma-separated string representation of the embedding
    """
    return ",".join(str(x) for x in embedding)


def string_to_embedding(embedding_str: str) -> List[float]:
    """
    Convert a comma-separated string to an embedding.

    Args:
        embedding_str: Comma-separated string representation of an embedding

    Returns:
        List of float values representing the embedding
    """
    return [float(x) for x in embedding_str.split(",")]


def calculate_text_hash(text: str) -> str:
    """
    Calculate MD5 hash of text content.

    Args:
        text: Text to hash

    Returns:
        MD5 hash of the text
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embedding for text using litellm.

    Args:
        text: Text to embed
        model: Embedding model to use

    Returns:
        List of float values representing the embedding
    """
    assert isinstance(text, str), "Text must be a string"

    if not text:
        return []

    try:
        return litellm_embedding(
            model=model,
            input=text,
        ).data[
            0
        ]["embedding"]
    except ContextWindowExceededError:
        logging.info("Context window exceeded, retrying with half the text")
        return get_embedding(
            text[int(len(text) / 2) :], model
        )  # Retry with half the text
