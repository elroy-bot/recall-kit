"""
API route handlers for Recall Kit.

This module provides FastAPI route handlers that implement an OpenAI-compatible API
for chat completions with memory augmentation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import Depends, HTTPException
from litellm import ChatCompletionRequest, ModelResponse  # type: ignore

from recall_kit import RecallKit

from ..models.message import Message, MessageSet
from ..processors.memory import MemoryConsolidator
from ..protocols.base import StorageBackendProtocol

# Set up logging
logger = logging.getLogger(__name__)


def get_recall_kit() -> RecallKit:
    """Get the RecallKit instance."""
    raise NotImplementedError("Should be implemented in the main app")


def get_memory_consolidator() -> MemoryConsolidator:
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
        memory_consolidator.consolidate_memories(model=request.model)
        return response

    except Exception as e:
        logger.exception("Error creating chat completion")
        raise HTTPException(status_code=500, detail=str(e))


async def get_messages(
    recall_kit: RecallKit = Depends(get_recall_kit),
) -> List[Message]:
    """
    Get all messages.

    Args:
        recall_kit: The RecallKit instance

    Returns:
        List of all messages
    """
    try:
        messages = recall_kit.get_all_messages()
        return [
            MessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at.isoformat(),
                metadata=message.metadata,
            )
            for message in messages
        ]
    except Exception as e:
        logger.exception("Error getting messages")
        raise HTTPException(status_code=500, detail=str(e))


async def get_message(
    message_id: str,
    recall_kit: RecallKit = Depends(get_recall_kit),
) -> Message:
    """
    Get a message by ID.

    Args:
        message_id: The ID of the message to retrieve
        recall_kit: The RecallKit instance

    Returns:
        The message
    """
    try:
        message = recall_kit.get_message(message_id)
        if not message:
            raise HTTPException(
                status_code=404, detail=f"Message {message_id} not found"
            )
        else:
            return message

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting message {message_id}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_message_sets(
    storage: StorageBackendProtocol = Depends(get_storage),
) -> List[MessageSet]:
    """
    Get all message sets.

    Args:
        recall_kit: The RecallKit instance

    Returns:
        List of all message sets
    """
    try:
        # This is not implemented in the RecallKit class, so we'll need to get it from the storage
        return recall_kit.storage.get_all_message_sets()

    except Exception as e:
        logger.exception("Error getting message sets")
        raise HTTPException(status_code=500, detail=str(e))


async def get_active_message_set(
    recall_kit: RecallKit = Depends(get_recall_kit),
) -> MessageSetResponse:
    """
    Get the active message set.

    Args:
        recall_kit: The RecallKit instance

    Returns:
        The active message set
    """
    try:
        message_set = recall_kit.get_active_message_set()
        if not message_set:
            raise HTTPException(status_code=404, detail="No active message set found")

        return MessageSetResponse(
            id=message_set.id,
            message_ids=message_set.message_ids,
            active=message_set.active,
            created_at=message_set.created_at.isoformat(),
            metadata=message_set.metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting active message set")
        raise HTTPException(status_code=500, detail=str(e))


async def get_message_set(
    message_set_id: str,
    recall_kit: RecallKit = Depends(get_recall_kit),
) -> MessageSetResponse:
    """
    Get a message set by ID.

    Args:
        message_set_id: The ID of the message set to retrieve
        recall_kit: The RecallKit instance

    Returns:
        The message set
    """
    try:
        message_set = recall_kit.get_message_set(message_set_id)
        if not message_set:
            raise HTTPException(
                status_code=404, detail=f"Message set {message_set_id} not found"
            )
        else:
            return message_set
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting message set {message_set_id}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_messages_in_set(
    message_set_id: str,
    recall_kit: RecallKit = Depends(get_recall_kit),
    storage: StorageBackendProtocol = Depends(get_storage),
) -> List[Message]:
    """
    Get all messages in a message set.

    Args:
        message_set_id: The ID of the message set
        recall_kit: The RecallKit instance

    Returns:
        List of messages in the message set
    """
    try:
        message_set = recall_kit.get_message_set(message_set_id)
        if not message_set:
            raise HTTPException(
                status_code=404, detail=f"Message set {message_set_id} not found"
            )

        return recall_kit.get_messages_in_set(message_set_id)
        return [
            MessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at.isoformat(),
                metadata=message.metadata,
            )
            for message in messages
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting messages in set {message_set_id}")
        raise HTTPException(status_code=500, detail=str(e))


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
