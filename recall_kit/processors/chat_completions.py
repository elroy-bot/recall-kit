import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from litellm import ModelResponse  # type: ignore
from litellm import Type
from pydantic import BaseModel

from ..constants import ASSISTANT, ROLE, TOOL
from ..core import CompletionFunction


def get_completion(
    completion_fn: CompletionFunction,
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
    response = completion_fn(**args)
    return response


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
