import logging
import uuid
from functools import wraps
from typing import cast

from litellm import ModelResponse  # type: ignore

from ..constants import ASSISTANT, ROLE, TOOL
from ..protocols.base import CompletionFunction


def memory_augmented(arg1, arg2) -> CompletionFunction:
    def decorator(func: CompletionFunction) -> CompletionFunction:
        @wraps(func)
        def wrapper(**kwargs) -> ModelResponse:
            return func(**kwargs)

        return cast(CompletionFunction, wrapper)

    return cast(CompletionFunction, decorator)


def ensure_correct_tool_messages(
    completion_fn: CompletionFunction,
) -> CompletionFunction:
    """
    Decorator for completion functions to ensure correct tool message handling.

    This decorator wraps a completion function to preprocess the request and ensure
    that tool messages are properly formatted according to the API requirements.

    Args:
        completion_fn: The completion function to wrap

    Returns:
        Wrapped completion function that ensures correct tool message handling
    """

    @wraps(completion_fn)
    def wrapper(**kwargs) -> ModelResponse:
        if not kwargs.get("messages"):
            return completion_fn(**kwargs)
        messages = kwargs["messages"]

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
                    tool_call_id = current_msg.get(
                        "tool_call_id", f"c_{str(uuid.uuid4())}"
                    )

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

        kwargs["messages"] = processed_messages

        return completion_fn(**kwargs)

    return cast(CompletionFunction, wrapper)
