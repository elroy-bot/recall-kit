"""
Utility functions for embedding operations.

This module provides utility functions for working with embeddings,
including conversion between different formats and vector operations.
"""

from __future__ import annotations

from typing import List

import numpy as np


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
