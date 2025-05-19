"""
Protocols package for Recall Kit.

This package exports protocol definitions for various functions used in Recall Kit.
"""

from recall_kit.protocols.base import (
    AugmentFunction,
    CompletionFunction,
    EmbeddingFunction,
    FilterFunction,
    RerankFunction,
    RetrieveFunction,
    StorageBackendProtocol,
)

__all__ = [
    "AugmentFunction",
    "CompletionFunction",
    "EmbeddingFunction",
    "FilterFunction",
    "RerankFunction",
    "RetrieveFunction",
    "StorageBackendProtocol",
]
