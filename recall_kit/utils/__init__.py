"""
Utility functions for Recall Kit.

This module provides utility functions for various operations in Recall Kit,
including embedding and completion operations.
"""


# Import completion utilities
from .completion import augment_with_memories, extract_content_from_response

# Import embedding utilities
from .embedding import bytes_to_embedding, calculate_similarity, embedding_to_bytes

__all__ = [
    # Embedding utilities
    "calculate_similarity",
    "embedding_to_bytes",
    "bytes_to_embedding",
    # Completion utilities
    "extract_content_from_response",
    "augment_with_memories",
    # Litellm types
]
