"""
Core data structures and functionality for Recall Kit.

This module contains the main classes and functions for working with memories,
including the RecallKit class.
"""

# search memory should:
# retrieve, filter, rerank, consider whether fetching more sources is needed


from __future__ import annotations

import uuid
from functools import partial
from typing import Deque, List, Optional, Tuple, TypeVar, Unpack

from litellm import AllMessageValues, ChatCompletionRequest
from litellm.types.utils import ModelResponse
from litellm.utils import token_counter
from toolz import pipe

from recall_kit.models import MessageSet
from recall_kit.plugins import registry

from .constants import CONTENT, MESSAGES, MODEL, ROLE, TOOL
from .models.pydantic_models import SourceMetadata
from .models.sql_models import Memory
from .processors.memory import MemoryConsolidator
from .services.embedding import EmbeddingService
from .services.memory import MemoryService
from .services.message_storage import MessageStorageService
from .utils.completion import extract_content_from_response
from .utils.messaging import to_tool_message, to_user_message

# Type variable for the RecallKit class
T = TypeVar("T", bound="RecallKit")


# Default functions have been moved to DefaultPlugin


class RecallKit:
    """
    Main class for working with memories.

    This class provides methods for creating, storing, retrieving, and
    consolidating memories.
    """

    from .protocols.base import (
        AugmentFunction,
        CompletionFunction,
        EmbeddingFunction,
        FilterFunction,
        RerankFunction,
        RetrieveFunction,
        StorageBackendProtocol,
    )

    def __init__(
        self,
        user_token: str,
        embedding_model: Optional[str] = None,
        token_limit: int = 10000,
        storage: Optional[StorageBackendProtocol] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        completion_fn: Optional[CompletionFunction] = None,
        retrieve_fn: Optional[RetrieveFunction] = None,
        filter_fn: Optional[FilterFunction] = None,
        rerank_fn: Optional[RerankFunction] = None,
        augment_fn: Optional[AugmentFunction] = None,
    ):
        """
        Initialize a new RecallKit instance.

        Args:
            storage: Storage backend instance
            embedding_fn: Function that takes a list of texts and returns a list of embedding vectors
            completion_fn: Function that takes a model name, messages, and optional parameters and returns a completion
            retrieve_fn: Function for retrieving memories
            filter_fn: Function for filtering memories
            rerank_fn: Function for reranking memories
            augment_fn: Function for augmenting requests with memories
        """
        self.token_limit = token_limit

        self.embedding_model = embedding_model or "text-embedding-3-small"

        # Set up storage backend
        self.storage = storage or registry.get_storage_backend("default")

        self.user_id: int = self.storage.get_user_by_token(
            user_token
        ) or self.storage.create_user(user_token)

        # Set up embedding and completion functions
        self.embedding_fn = embedding_fn or registry.get_embedding_fn("default")
        self.completion_fn = completion_fn or registry.get_completion_fn("default")

        # Set up custom functions
        self.retrieve_fn = retrieve_fn or registry.get_retrieve_fn("default")
        self.filter_fn = filter_fn or registry.get_filter_fn("default")
        self.rerank_fn = rerank_fn or registry.get_rerank_fn("default")
        self.augment_fn = augment_fn or registry.get_augment_fn("default")

        self.message_storage_service = MessageStorageService(
            storage=self.storage, user_id=self.user_id
        )

        self.memory_consolidator = MemoryConsolidator(
            embedding_model=self.embedding_model,
            storage=self.storage,
            completion_fn=self.completion_fn,
            embedding_fn=self.embedding_fn,
        )

        self.embedding_service = EmbeddingService(
            self.storage, self.embedding_fn, self.embedding_model
        )

        self.memory_store = MemoryService(
            filter_fn=self.filter_fn,
            rerank_fn=self.rerank_fn,
            storage=self.storage,
            embedding_model=self.embedding_model,
            embedding_fn=self.embedding_fn,
        )

    def remember_dropped_messages(
        self, completion_model: str, messages: List[AllMessageValues]
    ) -> Optional[Memory]:
        """
        Remember dropped messages by creating memories from them.

        Args:
            messages: List of messages to remember
        """

        if not messages:
            return

        messages = messages + [
            to_user_message(
                "The preceeding messages have been dropped from conversation context. Please summarize them in a single message."
            )
        ]

        summary = extract_content_from_response(
            self.completion_fn(model=completion_model, messages=messages)
        )

        message_set_id = self.message_storage_service.create_inactive_message_set(
            messages
        )

        return self.memory_store.create_memory(
            source_metadata=[
                SourceMetadata(
                    source_type=MessageSet.__name__, source_id=message_set_id
                )
            ],
            text=summary,
            user_id=self.user_id,
        )

    def insert_synthetic_tool_call(
        self, messages: List[AllMessageValues], tool_name: str, tool_content: str
    ) -> List[AllMessageValues]:
        """
        Insert a synthetic tool call into the messages. Tool call will be inserted after the first assistant message, and that assistant message will be updated to include the tool call.

        Args:
            messages: List of messages to insert the tool call into
            tool_name: Name of the tool to call
            tool_content: Content to pass to the tool
        """

        from recall_kit.constants import ASSISTANT

        if not messages:
            return messages

        # Find the first assistant message
        for i, msg in enumerate(messages):
            if msg[ROLE] == ASSISTANT:
                # Create a new tool message

                # Update the assistant message to include the tool call

                tool_call_id = f"c_{str(uuid.uuid4())}"
                messages[i]["tool_calls"] = [  # type: ignore
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ]

                tool_message = to_tool_message(
                    content=tool_content, tool_call_id=tool_call_id
                )
                messages.insert(i + 1, tool_message)
                return messages
        raise ValueError("No assistant message found to insert tool call into.")

    def completion(self, **request: Unpack[ChatCompletionRequest]) -> ModelResponse:
        request_with_stored_messages = self.message_storage_service.get_stored_messages(
            request
        )

        token_count = token_counter(
            request_with_stored_messages[MODEL],
            messages=request_with_stored_messages[MESSAGES],
        )

        if token_count > self.token_limit:
            kept_messages, dropped_messages = self.compress_messages(
                model=request_with_stored_messages[MODEL],
                messages=request_with_stored_messages[MESSAGES],
            )

            self.message_storage_service.set_conversation_messages(kept_messages)
            new_memory = self.remember_dropped_messages(
                request[MODEL], dropped_messages
            )

            if new_memory:
                new_messages = self.insert_synthetic_tool_call(
                    kept_messages, "get_convo_summary", new_memory.content
                )
                self.message_storage_service.set_conversation_messages(new_messages)
            else:
                new_messages = kept_messages
            request_with_stored_messages[MESSAGES] = new_messages

        return pipe(
            self.retrieve_fn(
                self.storage,
                self.embedding_model,
                self.embedding_fn,
                request_with_stored_messages,
            ),
            partial(self.filter_fn, request_with_stored_messages),
            partial(self.rerank_fn, request_with_stored_messages),
            partial(self.augment_fn, request_with_stored_messages),
            lambda r: self.completion_fn(**r),
        )  # type: ignore

    def compress_messages(
        self,
        model: str,
        messages: List[AllMessageValues],
    ) -> Tuple[List[AllMessageValues], List[AllMessageValues]]:
        """
        Compress messages to fit within the context window, by summarizing earlier messages.

        Creates memories from compressed messages that are dropped from the context.
        Appends a summary of dropped messages to the earliest kept assistant message.
        Creates a new message set with the kept messages and marks the old one inactive.

        Args:
            messages: List of messages to compress
            model: Model name to use for token counting
            target_token_count: Target number of tokens to keep

        Returns:
           Compressed messages
        """
        from collections import deque

        from litellm import token_counter  # type: ignore

        from recall_kit.constants import ASSISTANT, SYSTEM

        if not messages:
            return ([], [])

        kept_messages: Deque[AllMessageValues] = deque()
        dropped_messages: List[AllMessageValues] = []

        if messages[0][ROLE] == SYSTEM:
            system_message = messages[0]
            messages = messages[1:]  # Remove the system message from the main list
            current_token_count = token_counter(model=model, messages=[system_message])
        else:
            current_token_count = 0

        # Process messages in reverse order (newest first)
        for msg in reversed(messages):
            msg_content = str(msg.get(CONTENT, ""))

            # Calculate tokens for this message
            msg_token_count = token_counter(model=model, text=msg_content)

            # Check if we need to keep this message due to tool calls
            if (
                len(kept_messages) > 0
                and kept_messages[0][ROLE] == TOOL
                and msg[ROLE] == ASSISTANT
            ):
                # If the last message kept was a tool call, we must keep the corresponding assistant message
                kept_messages.appendleft(msg)
                current_token_count += msg_token_count
                continue

            # Check if we've exceeded our token budget
            if current_token_count + msg_token_count > self.token_limit / 2:
                # This message would put us over the limit
                dropped_messages.append(msg)
                continue

            # If we get here, keep the message
            kept_messages.appendleft(msg)
            current_token_count += msg_token_count

        return (list(kept_messages), dropped_messages)
