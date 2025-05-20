"""
Default plugin implementation for Recall Kit.

This module provides the default implementations of various functions used by Recall Kit,
including retrieval, filtering, reranking, augmentation, embedding, and completion.
"""

from __future__ import annotations

from typing import List, Unpack

from litellm import ChatCompletionRequest, ModelResponse  # type: ignore
from litellm.types.utils import EmbeddingResponse
from toolz import pipe
from toolz.curried import filter, map, take

from ..constants import ASSISTANT, CONTENT, ROLE, USER
from ..models.memory import Memory
from ..protocols.base import EmbeddingFunction, StorageBackendProtocol
from ..storage.sqlite import SQLiteBackend


class DefaultPlugin:
    """Default plugin for Recall Kit."""

    @staticmethod
    def retrieve(
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        request: ChatCompletionRequest,
    ) -> List[Memory]:
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
    def filter(request: ChatCompletionRequest, memories: List[Memory]) -> List[Memory]:
        """
        Default filter function.

        Args:
            memory: The memory to filter
            request: The original request

        Returns:
            True if the memory should be included, False otherwise
        """
        # Keep memories with relevance > 0.7
        answers = []

        for memory in memories:
            if not hasattr(memory, "relevance"):
                raise ValueError("Memory object does not have a 'relevance' attribute.")
            elif getattr(memory, "relevance") > 0.7:
                answers.append(memory)
        return answers

    @staticmethod
    def rerank(request: ChatCompletionRequest, memories: List[Memory]) -> List[Memory]:
        """
        Default rerank function.

        Args:
            memories: The list of memories to rerank
            request: The original request

        Returns:
            Reranked list of memories
        """
        # Sort by relevance

        if not hasattr(memories, "relevance"):
            raise ValueError("Memory object does not have a 'relevance' attribute.")
        return sorted(memories, key=lambda m: getattr(m, "relevance"), reverse=True)

    @staticmethod
    def augment(
        request: ChatCompletionRequest,
        memories: List[Memory],
    ) -> ChatCompletionRequest:
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

        from recall_kit.utils.completion import augment_with_memories

        # Format memories as text
        memories_text = "\n".join([f"- {getattr(m, 'text', str(m))}" for m in memories])

        # Augment the request with memories
        return augment_with_memories(request, memories_text)

    @staticmethod
    def embedding_fn(model: str, text: str) -> List[float]:
        """
        Default embedding function using litellm.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """

        from litellm import embedding as litellm_embedding

        resp = litellm_embedding(model=model, texts=[text])
        assert isinstance(resp, EmbeddingResponse)

        return resp.data[0]["embedding"]

    @staticmethod
    def completion_fn(
        **request: Unpack[ChatCompletionRequest],
    ) -> ModelResponse:
        """
        Default completion function using litellm.

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

        from litellm import completion as litellm_completion

        resp = litellm_completion(**request)  # type: ignore

        assert isinstance(resp, ModelResponse), "Response is not of type ModelResponse"
        return resp


# Import registration functions from registry module
# This needs to be after the class definition to avoid circular imports
from .registry import registry

# SQLiteBackend is already imported above

# Register the default embedding function
registry.register_embedding_fn(
    DefaultPlugin.embedding_fn,
    "default",
    aliases=["text-embedding-3-small", "text-embedding-ada-002"],
)

# Register the default completion function
registry.register_completion_fn(
    DefaultPlugin.completion_fn,
    "default",
    aliases=["gpt-3.5-turbo", "gpt-4", "claude-3-opus-20240229"],
)

# Register the default callback functions
registry.register_retrieve_fn(DefaultPlugin.retrieve, "default")

registry.register_filter_fn(DefaultPlugin.filter, "default")

registry.register_rerank_fn(DefaultPlugin.rerank, "default")

registry.register_augment_fn(DefaultPlugin.augment, "default")

registry.register_storage_backend(
    SQLiteBackend, "default", aliases=["sqlite", "sqlite3"]
)
