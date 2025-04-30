"""
Default plugin implementation for Recall Kit.

This module provides the default implementations of various functions used by Recall Kit,
including retrieval, filtering, reranking, augmentation, embedding, and completion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from litellm import ModelResponse

from .storage import SQLiteBackend
from .constants import ASSISTANT, CONTENT, ROLE, USER
from .models import Memory
from .core import EmbeddingFunction, StorageBackendProtocol

from toolz import pipe
from toolz.curried import map, filter, take


class DefaultPlugin:
    """Default plugin for Recall Kit."""

    @staticmethod
    def retrieve(storage: StorageBackendProtocol, embedding_fn: EmbeddingFunction, request: Any) -> List[Memory]:
        """
        Default retrieve function.

        Args:
            storage: The storage backend to use
            embedding_fn: The embedding function to use
            request: The original request (e.g., ChatCompletionRequest)

        Returns:
            List of relevant memories
        """
        return pipe(
            request.get("messages", []),
            filter(lambda m: m.get(ROLE) in [USER, ASSISTANT]),
            map(lambda m: f"{m.get(ROLE)}: {m.get(CONTENT)}"),
            take(3),
            list,
            "\n".join,
            embedding_fn,
            take(1),
            lambda e: storage.search_memories(e, limit=5),
            list,
        )

    @staticmethod
    def filter(memories: List[Memory], request: Any) -> bool:
        """
        Default filter function.

        Args:
            memory: The memory to filter
            request: The original request

        Returns:
            True if the memory should be included, False otherwise
        """
        # Keep memories with relevance > 0.7
        return [m for m in memories if getattr(m, "relevance", 0) > 0.7]

    @staticmethod
    def rerank(memories: List[Memory], request: Any) -> List[Memory]:
        """
        Default rerank function.

        Args:
            memories: The list of memories to rerank
            request: The original request

        Returns:
            Reranked list of memories
        """
        # Sort by relevance
        return sorted(memories, key=lambda m: getattr(m, "relevance", 0), reverse=True)

    @staticmethod
    def augment(memories: List[Memory], request: Any) -> Any:
        """
        Default augment function.

        Args:
            memories: The list of memories to include
            request: The original request to augment

        Returns:
            Augmented request
        """
        if not memories:
            return request

        # Create a copy of the request to avoid modifying the original
        augmented_request = dict(request)

        memory_context = "Relevant memories:\n" + "\n".join([f"- {getattr(m, 'text', str(m))}" for m in memories])

        if "messages" in augmented_request:
            # Find system message if it exists
            system_msg_idx = next((i for i, msg in enumerate(augmented_request["messages"])
                                if msg.get("role") == "system"), None)

            if system_msg_idx is not None:
                # Append to existing system message
                augmented_request["messages"][system_msg_idx]["content"] += f"\n\n{memory_context}"
            else:
                # Insert new system message at the beginning
                augmented_request["messages"].insert(0, {
                    "role": "system",
                    "content": memory_context
                })

        return augmented_request

    @staticmethod
    def embedding_fn(text: str) -> List[float]:
        """
        Default embedding function using litellm.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        from litellm import embedding as litellm_embedding, ContextWindowExceededError

        assert isinstance(text, str), "Text must be a string"

        if not text:
            return []
        try:
            return litellm_embedding(
                model="text-embedding-3-small",
                input=text,
            ).data[0]['embedding']
        except ContextWindowExceededError:
            logging.info("Context window exceeded, retrying with half the text")
            return DefaultPlugin.embedding_fn(text[int(len(text) / 2) :])  # Retry with half the text
        except Exception as e:
            return [[0.0] * 1536]

    @staticmethod
    def completion_fn(
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Default completion function using litellm.

        Args:
            model: The model to use for completion
            messages: List of messages in the conversation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            additional_args: Additional arguments to pass to litellm

        Returns:
            Completion response
        """
        from litellm import completion as litellm_completion, ContextWindowExceededError

        # Prepare arguments
        args = {
            "model": model,
            "messages": messages
        }

        # Add optional arguments if provided
        if max_tokens is not None:
            args["max_tokens"] = max_tokens

        if temperature is not None:
            args["temperature"] = temperature

        # Handle user parameter
        if additional_args and "user" in additional_args:
            # Keep the user parameter in args
            user_token = additional_args.get("user")
            args["user"] = user_token

        # Add any other additional arguments
        if additional_args:
            args.update(additional_args)

        try:
            # Get completion from litellm
            response = litellm_completion(**args)
            return response
        except ContextWindowExceededError:
            raise
        except Exception as e:
            # Return error response
            return {
                "choices": [
                    {
                        "finish_reason": "error",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error generating completion: {str(e)}"
                        }
                    }
                ],
                "created": None,
                "model": model,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }


# Import registration functions from plugins module
# This needs to be after the class definition to avoid circular imports
from .plugins import (
    register_embedding_fn,
    register_completion_fn,
    register_retrieve_fn,
    register_filter_fn,
    register_rerank_fn,
    register_augment_fn,
    register_storage_backend
)

# Register the default embedding function
register_embedding_fn(
    DefaultPlugin.embedding_fn,
    "default",
    aliases=["text-embedding-3-small", "text-embedding-ada-002"]
)

# Register the default completion function
register_completion_fn(
    DefaultPlugin.completion_fn,
    "default",
    aliases=["gpt-3.5-turbo", "gpt-4", "claude-3-opus-20240229"]
)

# Register the default callback functions
register_retrieve_fn(
    DefaultPlugin.retrieve,
    "default"
)

register_filter_fn(
    DefaultPlugin.filter,
    "default"
)

register_rerank_fn(
    DefaultPlugin.rerank,
    "default"
)

register_augment_fn(
    DefaultPlugin.augment,
    "default"
)

register_storage_backend(
    SQLiteBackend(),
    "default",
    aliases=["sqlite", "sqlite3"]
)
