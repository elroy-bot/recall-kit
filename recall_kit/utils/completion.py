"""
Utility functions for completion operations.

This module provides utility functions for working with completions,
including processing requests and responses.
"""

from __future__ import annotations

from typing import Any, Dict

from litellm import ModelResponse  # type: ignore

from ..constants import ROLE


def extract_content_from_response(response: ModelResponse) -> str:
    """
    Extract content from a completion response.

    Args:
        response: The completion response

    Returns:
        The extracted content as a string
    """
    if hasattr(response, "choices") and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):  # type: ignore
            return choice.message.content  # type: ignore

    # Fallback for different response structures
    try:
        return response.choices[0].message.content  # type: ignore
    except (AttributeError, IndexError):
        pass

    try:
        return response.choices[0].text  # type: ignore
    except (AttributeError, IndexError):
        pass

    # If we can't extract content, return empty string
    return ""


def augment_with_memories(
    request: Dict[str, Any], memories_text: str
) -> Dict[str, Any]:
    """
    Augment a request with memories.

    Args:
        request: The original request
        memories_text: Text representation of memories

    Returns:
        Augmented request
    """
    if not memories_text:
        return request

    # Create a copy of the request to avoid modifying the original
    augmented_request = dict(request)

    memory_context = f"Relevant memories:\n{memories_text}"

    if "messages" in augmented_request:
        # Find system message if it exists
        system_msg_idx = next(
            (
                i
                for i, msg in enumerate(augmented_request["messages"])
                if msg.get(ROLE) == "system"
            ),
            None,
        )

        if system_msg_idx is not None:
            # Append to existing system message
            augmented_request["messages"][system_msg_idx][
                "content"
            ] += f"\n\n{memory_context}"
        else:
            # Insert new system message at the beginning
            augmented_request["messages"].insert(
                0, {ROLE: "system", "content": memory_context}
            )

    return augmented_request
