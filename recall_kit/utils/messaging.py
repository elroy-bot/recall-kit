# File which creates constructors for relevant litellm.AllMessageValues classes:
# AllMessageValues = Union[
#     ChatCompletionUserMessage,
#     ChatCompletionAssistantMessage,
#     ChatCompletionToolMessage,
#     ChatCompletionSystemMessage,
#     ChatCompletionFunctionMessage,
#     ChatCompletionDeveloperMessage,
# ]

from typing import Iterable, List, Optional, Union

from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionDeveloperMessage,
    ChatCompletionFunctionMessage,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    OpenAIMessageContent,
)

from ..constants import CONTENT, ROLE, SYSTEM, USER


def to_user_message(
    content: OpenAIMessageContent,
) -> ChatCompletionUserMessage:
    """
    Create a user message for chat completions.

    Args:
        content: The content of the message, either as a string or a list of content blocks

    Returns:
        A ChatCompletionUserMessage dictionary
    """
    return {ROLE: USER, CONTENT: content}


def to_assistant_message(
    content: Optional[
        Union[
            str, Iterable[Union[ChatCompletionTextObject, ChatCompletionThinkingBlock]]
        ]
    ] = None,
    name: Optional[str] = None,
    tool_calls: Optional[List[ChatCompletionAssistantToolCall]] = None,
    function_call: Optional[ChatCompletionToolCallFunctionChunk] = None,
    reasoning_content: Optional[str] = None,
    thinking_blocks: Optional[List[ChatCompletionThinkingBlock]] = None,
) -> ChatCompletionAssistantMessage:
    """
    Create an assistant message for chat completions.

    Args:
        content: The content of the message, either as a string or a list of content blocks
        name: Optional name for the assistant
        tool_calls: Optional list of tool calls made by the assistant
        function_call: Optional function call made by the assistant (legacy)
        reasoning_content: Optional reasoning content
        thinking_blocks: Optional thinking blocks

    Returns:
        A ChatCompletionAssistantMessage dictionary
    """
    message: ChatCompletionAssistantMessage = {ROLE: "assistant"}

    if content is not None:
        message[CONTENT] = content

    if name is not None:
        message["name"] = name

    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    if function_call is not None:
        message["function_call"] = function_call

    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    if thinking_blocks is not None:
        message["thinking_blocks"] = thinking_blocks

    return message


def to_tool_message(
    content: Union[str, Iterable[ChatCompletionTextObject]],
    tool_call_id: str,
) -> ChatCompletionToolMessage:
    """
    Create a tool message for chat completions.

    Args:
        content: The content of the message, either as a string or a list of content blocks
        tool_call_id: The ID of the tool call this message is responding to

    Returns:
        A ChatCompletionToolMessage dictionary
    """
    return {
        ROLE: "tool",
        CONTENT: content,
        "tool_call_id": tool_call_id,
    }


def to_system_message(
    content: Union[str, List],
    name: Optional[str] = None,
) -> ChatCompletionSystemMessage:
    """
    Create a system message for chat completions.

    Args:
        content: The content of the message, either as a string or a list
        name: Optional name for the system

    Returns:
        A ChatCompletionSystemMessage dictionary
    """
    message: ChatCompletionSystemMessage = {ROLE: SYSTEM, CONTENT: content}

    if name is not None:
        message["name"] = name

    return message


def function_message(
    name: str,
    tool_call_id: str,
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]] = None,
) -> ChatCompletionFunctionMessage:
    """
    Create a function message for chat completions.

    Args:
        name: The name of the function
        tool_call_id: ID of the tool call this message is responding to
        content: Optional content of the message, either as a string or a list of content blocks

    Returns:
        A ChatCompletionFunctionMessage dictionary
    """
    return {
        ROLE: "function",
        "name": name,
        "tool_call_id": tool_call_id,
        CONTENT: content,
    }


def developer_message(
    content: Union[str, List],
    name: Optional[str] = None,
) -> ChatCompletionDeveloperMessage:
    """
    Create a developer message for chat completions.

    Args:
        content: The content of the message, either as a string or a list
        name: Optional name for the developer

    Returns:
        A ChatCompletionDeveloperMessage dictionary
    """
    message: ChatCompletionDeveloperMessage = {ROLE: "developer", CONTENT: content}

    if name is not None:
        message["name"] = name

    return message
