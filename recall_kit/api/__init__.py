"""
API module for Recall Kit.

This module exports the API components for the Recall Kit server.
"""

from recall_kit.api.routes import (  # Models; Route handlers
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    Message,
    MessageResponse,
    MessageSetResponse,
    create_chat_completion,
    get_active_message_set,
    get_message,
    get_message_set,
    get_message_sets,
    get_messages,
    get_messages_in_set,
    get_recall_kit,
    list_models,
)

__all__ = [
    # Models
    "Message",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "MessageResponse",
    "MessageSetResponse",
    "ChatCompletionResponse",
    # Route handlers
    "get_recall_kit",
    "create_chat_completion",
    "get_messages",
    "get_message",
    "get_message_sets",
    "get_active_message_set",
    "get_message_set",
    "get_messages_in_set",
    "list_models",
]
