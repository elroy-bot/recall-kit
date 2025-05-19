"""
Storage package for Recall Kit.

This package provides storage backends for storing and retrieving memories,
including SQLite backend and vector storage utilities.
"""

from recall_kit.models import Embedding, Memory, Message, MessageSet, User
from recall_kit.storage.sqlite import SQLiteBackend

__all__ = [
    # Base storage classes and utilities
    "Embedding",
    "Memory",
    "MessageSet",
    "Message",
    "User",
    # SQLite backend
    "SQLiteBackend",
]
