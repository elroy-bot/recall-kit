"""
Utility functions for completion operations.

This module provides utility functions for working with completions,
including processing requests and responses.
"""

from __future__ import annotations

from litellm import ChatCompletionRequest, ModelResponse  # type: ignore

from ..constants import CONTENT, ROLE, SYSTEM


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
        if hasattr(choice, "message") and hasattr(choice.message, CONTENT):  # type: ignore
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
    raise ValueError("Could not extract content from response.")


def augment_with_memories(
    request: ChatCompletionRequest, memories_text: str
) -> ChatCompletionRequest:
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

    messages = request["messages"]

    # Create a copy of the request to avoid modifying the original
    augmented_request = dict(request)

    memory_context = f"Relevant memories:\n{memories_text}"

    if "messages" in augmented_request:
        # Find system message if it exists
        system_msg_idx = next(
            (i for i, msg in enumerate(messages) if msg.get(ROLE) == SYSTEM),
            None,
        )

        if system_msg_idx is not None:
            # Append to existing system message
            messages[system_msg_idx][CONTENT] += f"\n\n{memory_context}"  # type: ignore
        else:
            # Insert new system message at the beginning
            messages.insert(0, {ROLE: SYSTEM, CONTENT: memory_context})

    request["messages"] = messages

    return request
