"""
OpenAI-compatible API server for Recall Kit.

This module provides a FastAPI server that implements an OpenAI-compatible API
for chat completions with memory augmentation.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from litellm import AllMessageValues, ModelResponse  # type: ignore

from recall_kit import RecallKit
from recall_kit.models import MessageSet

from .api.routes import (
    create_chat_completion,
    get_active_message_set,
    get_memory_consolidator,
    get_message,
    get_message_set,
    get_messages_in_set,
    get_recall_kit,
    list_models,
)

# Set up logging
logger = logging.getLogger(__name__)


def create_app(
    recall: RecallKit,
) -> FastAPI:
    """
    Create a FastAPI app for the Recall Kit server.

    Args:
        memory_db_path: Path to the memory database
        auto_consolidate: Whether to automatically consolidate memories
        model: The model to use for completions and consolidation

    Returns:
        A FastAPI app
    """
    app = FastAPI(
        title="Recall Kit API",
        description="OpenAI-compatible API for chat completions with memory augmentation",
        version="0.1.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Override the get_recall_kit dependency
    app.dependency_overrides[get_recall_kit] = lambda: recall
    app.dependency_overrides[
        get_memory_consolidator
    ] = lambda: recall.memory_consolidator

    # Import litellm here to avoid requiring it for non-server use
    try:
        pass
    except ImportError:
        raise ImportError(
            "litellm is required for the server. "
            "Install it with: pip install litellm"
        )

    # Register routes
    app.post("/v1/chat/completions", response_model=ModelResponse)(
        create_chat_completion
    )

    app.get("/v1/messages/{message_id}", response_model=AllMessageValues)(get_message)
    app.get("/v1/message-sets/active", response_model=MessageSet)(
        get_active_message_set
    )
    app.get("/v1/message-sets/{message_set_id}", response_model=MessageSet)(
        get_message_set
    )
    app.get(
        "/v1/message-sets/{message_set_id}/messages",
        response_model=list[AllMessageValues],
    )(get_messages_in_set)
    app.get("/v1/models")(list_models)

    return app
