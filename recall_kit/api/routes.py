"""
API route handlers for Recall Kit.

This module provides FastAPI route handlers that implement an OpenAI-compatible API
for chat completions with memory augmentation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from litellm import AllMessageValues, ChatCompletionRequest
from litellm.types.utils import ModelResponse

from ..core import RecallKit  # type: ignore
from ..models import MessageSet
from ..processors.memory import MemoryConsolidator
from ..protocols.base import StorageBackendProtocol

# Set up logging
logger = logging.getLogger(__name__)


def get_recall_kit() -> RecallKit:
    """Get the RecallKit instance."""
    raise NotImplementedError("Should be implemented in the main app")


def get_memory_consolidator() -> MemoryConsolidator:
    raise NotImplementedError("Should be implemented in the main app")


def get_storage() -> StorageBackendProtocol:
    """Get the storage backend."""
    raise NotImplementedError("Should be implemented in the main app")


async def create_chat_completion(
    request: ChatCompletionRequest,
    recall_kit: RecallKit = Depends(get_recall_kit),
    memory_consolidator: MemoryConsolidator = Depends(get_memory_consolidator),
) -> ModelResponse:
    """
    Create a chat completion with memory augmentation.

    Args:
        request: The chat completion request
        recall_kit: The RecallKit instance

    Returns:
        The chat completion response
    """
    try:
        # Use the chat_completion method which handles the entire pipeline
        response = recall_kit.completion(**request)

        # Always perform auto-consolidation after each chat completion
        # Use LLM-driven consolidation
        memory_consolidator.consolidate_memories(completion_model=request["model"])
        return response

    except Exception as e:
        logger.exception("Error creating chat completion")
        raise HTTPException(status_code=500, detail=str(e))


async def get_message(
    message_id: int,
    storage: StorageBackendProtocol = Depends(get_storage),
) -> AllMessageValues:
    """
    Get a message by ID.

    Args:
        message_id: The ID of the message to retrieve
        recall_kit: The RecallKit instance

    Returns:
        The message
    """
    message = storage.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
    else:
        return message


async def get_active_message_set(
    storage: StorageBackendProtocol = Depends(get_storage),
) -> Optional[MessageSet]:
    """
    Get the active message set.

    Args:
        recall_kit: The RecallKit instance

    Returns:
        The active message set
    """
    return storage.get_active_message_set()


async def get_message_set(
    message_set_id: int,
    storage: StorageBackendProtocol = Depends(get_storage),
) -> MessageSet:
    """
    Get a message set by ID.

    Args:
        message_set_id: The ID of the message set to retrieve
        recall_kit: The RecallKit instance

    Returns:
        The message set
    """
    message_set = storage.get_message_set(message_set_id)
    if not message_set:
        raise HTTPException(
            status_code=404, detail=f"Message set {message_set_id} not found"
        )
    else:
        return message_set


async def get_messages_in_set(
    message_set_id: int,
    storage: StorageBackendProtocol = Depends(get_storage),
) -> List[AllMessageValues]:
    """
    Get all messages in a message set.

    Args:
        message_set_id: The ID of the message set
        recall_kit: The RecallKit instance

    Returns:
        List of messages in the message set
    """
    messages = storage.get_messages_in_set(message_set_id)
    if not messages:
        raise HTTPException(
            status_code=404, detail=f"Message set {message_set_id} not found"
        )
    return messages


async def list_models() -> Dict[str, Any]:
    """
    List available models.

    Returns:
        A list of available models
    """
    try:
        # Return a list of common models since litellm.model_list is not callable
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-4-turbo", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "text-embedding-3-small", "object": "model"},
                {"id": "text-embedding-3-large", "object": "model"},
            ],
        }
    except Exception:
        logger.exception("Error listing models")
        # Return a minimal response with common models
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-4-turbo", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
            ],
        }
