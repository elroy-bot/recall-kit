"""
Storage package for Recall Kit.

This package provides storage backends for storing and retrieving memories,
including SQLite backend and vector storage utilities.
"""

from recall_kit.storage.base import (
    Embedding,
    Memory,
    Message,
    MessageSet,
    User,
    parse_json_field,
    serialize_json_field,
)
from recall_kit.storage.sqlite import SQLiteBackend

__all__ = [
    # Base storage classes and utilities
    "Embedding",
    "Memory",
    "MessageSet",
    "Message",
    "User",
    "parse_json_field",
    "serialize_json_field",
    # SQLite backend
    "SQLiteBackend",
]
