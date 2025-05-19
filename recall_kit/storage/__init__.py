"""
Storage package for Recall Kit.

This package provides storage backends for storing and retrieving memories,
including SQLite backend and vector storage utilities.
"""

from recall_kit.storage.base import (
    EmbeddingTable,
    MemoryTable,
    MessageSetTable,
    MessageTable,
    UserTable,
    parse_json_field,
    serialize_json_field,
)
from recall_kit.storage.sqlite import SQLiteBackend
from recall_kit.storage.vector import (
    compute_text_hash,
    cosine_similarity,
    deserialize_embedding,
    embedding_to_string,
    filter_memories_by_threshold,
    rank_memories_by_relevance,
    serialize_embedding,
)

__all__ = [
    # Base storage classes and utilities
    "EmbeddingTable",
    "MemoryTable",
    "MessageSetTable",
    "MessageTable",
    "UserTable",
    "parse_json_field",
    "serialize_json_field",
    # SQLite backend
    "SQLiteBackend",
    # Vector utilities
    "compute_text_hash",
    "cosine_similarity",
    "deserialize_embedding",
    "embedding_to_string",
    "filter_memories_by_threshold",
    "rank_memories_by_relevance",
    "serialize_embedding",
]
