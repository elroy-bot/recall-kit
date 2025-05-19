"""
Utility functions for completion operations.

This module provides utility functions for working with completions,
including processing requests and responses.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Type, Union

from litellm import ModelResponse
from litellm import completion as litellm_completion
from pydantic import BaseModel

from ..constants import ASSISTANT, ROLE, TOOL


def process_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process messages to ensure proper tool message handling.

    This function ensures that tool messages have a preceding assistant message
    with tool_calls, which is required by many LLM APIs.

    Args:
        messages: List of messages to process

    Returns:
        Processed list of messages
    """
    processed_messages = []
    i = 0

    while i < len(messages):
        current_msg = messages[i]

        # Handle regular messages
        if current_msg.get(ROLE) != TOOL:
            processed_messages.append(current_msg)
            i += 1
            continue

        # Handle tool messages - they need a preceding assistant message with tool_calls
        if i > 0 and messages[i - 1].get(ROLE) == ASSISTANT:
            # Get the previous assistant message
            prev_assistant_msg = processed_messages[-1]

            # If the assistant message doesn't have tool_calls, add it
            if "tool_calls" not in prev_assistant_msg:
                # Generate a tool_call_id that's at most 40 characters
                # UUID is 36 chars, so we use a shorter prefix to stay under 40
                tool_call_id = current_msg.get("tool_call_id", f"c_{str(uuid.uuid4())}")

                # Add tool_calls to the assistant message
                prev_assistant_msg["tool_calls"] = [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": current_msg.get("metadata", {}).get(
                                "name", "get_information"
                            ),
                            "arguments": "{}",
                        },
                    }
                ]

                # Add tool_call_id to the tool message if it doesn't have one
                if "tool_call_id" not in current_msg:
                    current_msg["tool_call_id"] = tool_call_id

            processed_messages.append(current_msg)
        else:
            # If there's no preceding assistant message, log a warning and skip this tool message
            logging.warning(
                f"Tool message without preceding assistant message: {current_msg}"
            )

        i += 1

    return processed_messages


def get_completion(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    additional_args: Optional[Dict[str, Any]] = None,
) -> ModelResponse:
    """
    Get completion using litellm.

    Args:
        model: The model to use for completion
        messages: List of messages in the conversation
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        response_format: Format for the response
        additional_args: Additional arguments to pass to litellm

    Returns:
        Completion response
    """
    # Process messages to ensure proper tool message handling
    processed_messages = process_tool_messages(messages)

    # Prepare arguments
    args = {"model": model, "messages": processed_messages}

    # Add optional arguments if provided
    if max_tokens is not None:
        args["max_tokens"] = max_tokens

    if temperature is not None:
        args["temperature"] = temperature

    if response_format is not None:
        args["response_format"] = response_format

    # Handle user parameter
    if additional_args and "user" in additional_args:
        # Keep the user parameter in args
        user_token = additional_args.get("user")
        args["user"] = user_token

    # Add any other additional arguments
    if additional_args:
        args.update(additional_args)

    # Get completion from litellm
    response = litellm_completion(**args)
    return response


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
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content

    # Fallback for different response structures
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError):
        pass

    try:
        return response.choices[0].text
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
