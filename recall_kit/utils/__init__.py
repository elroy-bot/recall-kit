"""
Utility functions for Recall Kit.

This module provides utility functions for various operations in Recall Kit,
including embedding and completion operations.
"""

# Import completion utilities
from .completion import (
    augment_with_memories,
    extract_content_from_response,
    get_completion,
    process_tool_messages,
)

# Import embedding utilities
from .embedding import (
    bytes_to_embedding,
    calculate_similarity,
    calculate_text_hash,
    embedding_to_bytes,
    embedding_to_string,
    get_embedding,
    string_to_embedding,
)

__all__ = [
    # Embedding utilities
    "calculate_similarity",
    "embedding_to_bytes",
    "bytes_to_embedding",
    "embedding_to_string",
    "string_to_embedding",
    "calculate_text_hash",
    "get_embedding",
    # Completion utilities
    "process_tool_messages",
    "get_completion",
    "extract_content_from_response",
    "augment_with_memories",
]
