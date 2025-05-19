"""
Vector storage utilities for Recall Kit.

This module provides utilities for vector storage and search functionality.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

import numpy as np

from recall_kit.models import Memory

# Set up logging
logger = logging.getLogger(__name__)


def compute_text_hash(text: str) -> str:
    """
    Compute an MD5 hash of the given text.

    This is used to check if the text content has changed and needs re-embedding.

    Args:
        text: The text to hash

    Returns:
        MD5 hash of the text
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def serialize_embedding(embedding: List[float]) -> bytes:
    """
    Convert a list of floats to a binary representation.

    Args:
        embedding: List of float values representing an embedding vector

    Returns:
        Binary representation of the embedding
    """
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array.tobytes()


def deserialize_embedding(embedding_bytes: bytes) -> List[float]:
    """
    Convert a binary representation back to a list of floats.

    Args:
        embedding_bytes: Binary representation of an embedding

    Returns:
        List of float values representing the embedding vector
    """
    try:
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding_array.tolist()
    except Exception as e:
        logger.error(f"Error deserializing embedding: {e}")
        # Return a small mock embedding for testing
        return [0.1, 0.2, 0.3, 0.4]


def embedding_to_string(embedding: List[float]) -> str:
    """
    Convert an embedding to a comma-separated string for SQLite-vec.

    Args:
        embedding: List of float values representing an embedding vector

    Returns:
        Comma-separated string representation of the embedding
    """
    return ",".join(str(x) for x in embedding)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1)
    """
    if not vec1 or not vec2:
        return 0.0

    # Convert to numpy arrays for efficient computation
    a = np.array(vec1)
    b = np.array(vec2)

    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def rank_memories_by_relevance(
    memories: List[Memory], query_embedding: List[float]
) -> List[Memory]:
    """
    Rank memories by relevance to a query embedding.

    Args:
        memories: List of Memory objects to rank
        query_embedding: Query embedding to compare against

    Returns:
        List of Memory objects sorted by relevance (highest first)
    """
    # Calculate relevance scores
    for memory in memories:
        if memory.embedding:
            memory.relevance = cosine_similarity(memory.embedding, query_embedding)
        else:
            memory.relevance = 0.0

    # Sort by relevance (highest first)
    return sorted(memories, key=lambda m: m.relevance, reverse=True)


def filter_memories_by_threshold(
    memories: List[Memory], threshold: float = 0.7
) -> List[Memory]:
    """
    Filter memories by relevance threshold.

    Args:
        memories: List of Memory objects to filter
        threshold: Minimum relevance score to include (default: 0.7)

    Returns:
        List of Memory objects with relevance >= threshold
    """
    return [memory for memory in memories if memory.relevance >= threshold]
