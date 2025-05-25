"""
Core data structures and functionality for Recall Kit.

This module contains the main classes and functions for working with memories,
including the RecallKit class.
"""

# search memory should:
# retrieve, filter, rerank, consider whether fetching more sources is needed


from __future__ import annotations

import datetime
import json
from functools import partial
from typing import Deque, List, Optional, TypeVar, Unpack

from litellm import AllMessageValues, ChatCompletionRequest
from litellm.types.utils import ModelResponse
from toolz import pipe

from recall_kit.models import Message, MessageSet
from recall_kit.plugins import registry

from .constants import CONTENT, ROLE, TOOL
from .processors.memory import MemoryConsolidator
from .services.embedding import EmbeddingService
from .services.memory import MemoryService
from .services.message_storage import MessageStorageService
from .utils.messaging import to_tool_message

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
        self.embedding_model = embedding_model or "text-embedding-3-small"

        # Set up storage backend
        self.storage = storage or registry.get_storage_backend("default")

        # Set up embedding and completion functions
        self.embedding_fn = embedding_fn or registry.get_embedding_fn("default")
        self.completion_fn = completion_fn or registry.get_completion_fn("default")

        # Set up custom functions
        self.retrieve_fn = retrieve_fn or registry.get_retrieve_fn("default")
        self.filter_fn = filter_fn or registry.get_filter_fn("default")
        self.rerank_fn = rerank_fn or registry.get_rerank_fn("default")
        self.augment_fn = augment_fn or registry.get_augment_fn("default")

        self.message_storage_service = MessageStorageService(storage=self.storage)

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
            storage=self.storage,
            embedding_model=self.embedding_model,
            embedding_fn=self.embedding_fn,
        )

    def completion(self, **request: Unpack[ChatCompletionRequest]) -> ModelResponse:
        request_with_stored_messages = self.message_storage_service.get_stored_messages(
            request
        )

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

    def create_message_set(
        self,
        message_ids: List[int],
        active: bool = True,
        user_id: Optional[int] = None,
    ) -> int:
        """
        Create a new message set.

        Args:
            message_ids: List of message IDs in this set
            active: Whether this message set is active
            metadata: Additional metadata about the message set
            user_id: ID of the user who owns this message set (defaults to default user if not provided)

        Returns:
            The created MessageSet object
        """
        # If this is an active message set, deactivate all other message sets
        if active:
            self.storage.deactivate_all_message_sets()

        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        assert isinstance(user_id, int), "user_id must be an integer"

        message_set = MessageSet(
            message_ids_str=json.dumps(message_ids),
            active=active,
            user_id=user_id,
        )

        # Store the message set
        return self.storage.store_message_set(message_set)

    def compress_messages(
        self,
        model: str,
        messages: List[Message],
        target_token_count: int = 4000,
        max_message_age: Optional[datetime.timedelta] = None,
    ) -> List[AllMessageValues]:
        """
        Compress messages to fit within the context window, by summarizing earlier messages.

        Creates memories from compressed messages that are dropped from the context.
        Appends a summary of dropped messages to the earliest kept assistant message.
        Creates a new message set with the kept messages and marks the old one inactive.

        Args:
            messages: List of messages to compress
            model: Model name to use for token counting
            target_token_count: Target number of tokens to keep
            max_message_age: Maximum age of messages to keep (None means no limit)

        Returns:
           Compressed messages
        """
        import datetime
        from collections import deque

        from litellm import token_counter  # type: ignore

        from recall_kit.constants import ASSISTANT, SYSTEM

        if not messages:
            return []

        # Find system message if it exists
        system_messages = [msg for msg in messages if msg.data[ROLE] == SYSTEM]
        non_system_messages = [msg for msg in messages if msg.data[ROLE] != SYSTEM]

        # If no system message, we'll just work with all messages
        if system_messages:
            system_message = system_messages[0]
            current_token_count = token_counter(
                model=model, text=str(system_message.data.get(CONTENT, ""))
            )
        else:
            system_message = None
            current_token_count = 0

        kept_messages: Deque[Message] = deque()
        dropped_messages = []

        # Process messages in reverse order (newest first)
        for msg in reversed(non_system_messages):
            msg_content = str(msg.data.get(CONTENT, ""))

            # Calculate tokens for this message
            msg_token_count = token_counter(model=model, text=msg_content)

            # Check if we need to keep this message due to tool calls
            if (
                len(kept_messages) > 0
                and kept_messages[0].data[ROLE] == TOOL
                and msg.data[ROLE] == ASSISTANT
            ):
                # If the last message kept was a tool call, we must keep the corresponding assistant message
                kept_messages.appendleft(msg)
                current_token_count += msg_token_count
                continue

            # Check if we've exceeded our token budget
            if current_token_count + msg_token_count > target_token_count:
                # This message would put us over the limit
                dropped_messages.append(msg)
                continue

            # Check if the message is too old (if we have a max age)
            if max_message_age and msg.created_at:
                msg_created_at = msg.created_at
                if isinstance(msg_created_at, str):
                    msg_created_at = datetime.datetime.fromisoformat(msg_created_at)

                if (
                    msg_created_at
                    and msg_created_at < datetime.datetime.now() - max_message_age
                ):
                    dropped_messages.append(msg)
                    continue

            # If we get here, keep the message
            kept_messages.appendleft(msg)
            current_token_count += msg_token_count

        # Create memories from dropped messages
        if dropped_messages:
            # Create a consolidated memory from the dropped messages
            dropped_text = "\n".join(
                [
                    f"{msg[ROLE].upper()}: {msg.get('content', '')}"
                    for msg in dropped_messages
                ]
            )

            memory_text: str = (
                self.completion_fn(
                    model=model,
                    messages=[
                        {
                            ROLE: SYSTEM,
                            CONTENT: "You are a memory summarizer. Summarize the following messages.",
                        },
                        {
                            ROLE: ASSISTANT,
                            CONTENT: dropped_text,
                        },
                    ],
                    temperature=0.0,
                    max_tokens=target_token_count,
                )
                .choices[0]
                .message.content  # type: ignore
            )  # type: ignore

            memory = self.memory_store.create_memory(
                text=memory_text,
                title="Summary of dropped messages",
                source_metadata=[
                    {
                        "source_type": MessageSet.__name__,
                        "source_id": self.storage.get_active_message_set().id,  # type: ignore
                    },
                ],
            )

            # Find the earliest kept assistant message to append the summary
            earliest_assistant_msg: Optional[Message] = None
            for msg in kept_messages:
                if msg.data.get(ROLE) == ASSISTANT:
                    earliest_assistant_msg = msg
                    break

            # If we found an assistant message, add a tool call and insert a tool results message
            if earliest_assistant_msg:
                # Create a new tool message with the summary
                summary_content = f"[Context: {len(dropped_messages)} earlier messages were summarized: {memory.title}]"

                # Create a tool message to insert after the earliest assistant message
                import uuid

                # Use a shorter prefix to ensure ID is under 40 characters
                tool_call_id = f"c_{str(uuid.uuid4())}"

                # Add tool_calls to the assistant message
                if "tool_calls" not in earliest_assistant_msg.data:
                    data = earliest_assistant_msg.data
                    data["tool_calls"] = [  # type: ignore
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "get_summary", "arguments": "{}"},
                        }
                    ]

                    # Need to create a NEW assistnat message in this scenario, and drop the existing assistant message

                new_id = self.storage.store_message(
                    to_tool_message(content=summary_content, tool_call_id=tool_call_id)
                )
                tool_message = self.storage.get_message(new_id)

                # Find the index of the earliest assistant message
                earliest_assistant_idx = None
                for i, msg in enumerate(kept_messages):
                    if msg is earliest_assistant_msg:
                        earliest_assistant_idx = i
                        break

                # Insert the tool message after the earliest assistant message
                if earliest_assistant_idx is not None:
                    kept_messages.insert(earliest_assistant_idx + 1, tool_message)  # type: ignore

        # Construct the final message list
        result: List[Message] = list(kept_messages)
        if system_message:
            result.insert(0, system_message)

        message_ids = []

        for msg in result:
            if msg.id is None:
                message_ids.append(self.storage.store_message(msg.data))
            else:
                message_ids.append(msg.id)

        # If we have an existing message set, mark it inactive
        old_message_set = self.storage.get_active_message_set()

        if old_message_set:
            old_message_set.active = False
            self.storage.store_message_set(old_message_set)

            # Create a new active message set
            self.create_message_set(
                message_ids=message_ids,
                active=True,
            )

        return [msg.data for msg in result]
