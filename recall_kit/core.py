"""
Core data structures and functionality for Recall Kit.

This module contains the main classes and functions for working with memories,
including the RecallKit class.
"""

from __future__ import annotations

import datetime
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from litellm import ModelResponse, Type
from pydantic import BaseModel, Field

from recall_kit.managers import ChatManager, MemoryManager, MessageManager
from recall_kit.models import Memory, Message, MessageSet

from .constants import CONTENT, ROLE, USER


# Define type protocols for the callback functions
@runtime_checkable
class RetrieveFunction(Protocol):
    def __call__(
        self,
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        request: Any,
    ) -> List[Memory]:
        ...


@runtime_checkable
class FilterFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> bool:
        ...


@runtime_checkable
class RerankFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> List[Memory]:
        ...


@runtime_checkable
class AugmentFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> Any:
        ...


@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, text: str) -> List[float]:
        ...


@runtime_checkable
class CompletionFunction(Protocol):
    def __call__(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        additional_args: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        ...


@runtime_checkable
class StorageBackendProtocol(Protocol):
    def store_memory(self, memory: Any) -> None:
        ...

    def get_memory(self, memory_id: str) -> Optional[Any]:
        ...

    def get_all_memories(self) -> List[Any]:
        ...

    def search_memories(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Any]:
        ...

    def update_memory(self, memory: Any) -> None:
        ...

    def delete_memory(self, memory_id: str) -> bool:
        ...

    def store_message(self, message: Any) -> None:
        ...

    def get_message(self, message_id: str) -> Optional[Any]:
        ...

    def get_all_messages(self) -> List[Any]:
        ...

    def store_message_set(self, message_set: Any) -> None:
        ...

    def get_message_set(self, message_set_id: str) -> Optional[Any]:
        ...

    def get_active_message_set(self) -> Optional[Any]:
        ...

    def get_messages_in_set(self, message_set_id: str) -> List[Any]:
        ...

    def deactivate_all_message_sets(self) -> None:
        ...

    def get_all_message_sets(self) -> List[Any]:
        ...

    def create_user(self, token: str) -> int:
        ...

    def get_user_by_token(self, token: str) -> Optional[int]:
        ...

    def get_default_user_id(self) -> int:
        ...


# Type variable for the RecallKit class
T = TypeVar("T", bound="RecallKit")


# Default functions have been moved to DefaultPlugin


class RecallKit:
    """
    Main class for working with memories.

    This class provides methods for creating, storing, retrieving, and
    consolidating memories.
    """

    def __init__(
        self,
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        completion_fn: CompletionFunction,
        retrieve_fn: RetrieveFunction,
        filter_fn: FilterFunction,
        rerank_fn: RerankFunction,
        augment_fn: AugmentFunction,
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
        # Set up storage backend
        self.storage = storage

        # Set up embedding and completion functions
        self.embedding_fn = embedding_fn
        self.completion_fn = completion_fn

        # Set up custom functions
        self.retrieve_fn = retrieve_fn
        self.filter_fn = filter_fn
        self.rerank_fn = rerank_fn
        self.augment_fn = augment_fn

        # Initialize managers
        self.memory_manager = MemoryManager(self)
        self.message_manager = MessageManager(self)
        self.chat_manager = ChatManager(self)

    @classmethod
    def create(
        cls: Type[T],
        storage: Optional[StorageBackendProtocol] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        completion_fn: Optional[CompletionFunction] = None,
        retrieve_fn: Optional[RetrieveFunction] = None,
        filter_fn: Optional[FilterFunction] = None,
        rerank_fn: Optional[RerankFunction] = None,
        augment_fn: Optional[AugmentFunction] = None,
    ) -> T:
        """
        Create a RecallKit instance with optional parameters.

        Any parameters not provided will use default implementations from plugins.

        Args:
            storage: Storage backend instance (optional)
            embedding_fn: Function for text embeddings (optional)
            completion_fn: Function for LLM completions (optional)
            retrieve_fn: Function for retrieving memories (optional)
            filter_fn: Function for filtering memories (optional)
            rerank_fn: Function for reranking memories (optional)
            augment_fn: Function for augmenting requests with memories (optional)

        Returns:
            A new RecallKit instance
        """
        from recall_kit.plugins import (
            get_augment_fn,
            get_completion_fn,
            get_embedding_fn,
            get_filter_fn,
            get_rerank_fn,
            get_retrieve_fn,
            get_storage_backend,
        )

        return cls(
            storage=storage or get_storage_backend("default"),
            embedding_fn=embedding_fn or get_embedding_fn("default"),
            completion_fn=completion_fn or get_completion_fn("default"),
            retrieve_fn=retrieve_fn or get_retrieve_fn("default"),
            filter_fn=filter_fn or get_filter_fn("default"),
            rerank_fn=rerank_fn or get_rerank_fn("default"),
            augment_fn=augment_fn or get_augment_fn("default"),
        )

    # Memory-related methods delegated to MemoryManager
    def create_memory(
        self,
        text: str,
        title: Optional[str] = None,
        source_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> Memory:
        """
        Create a new memory from text.

        Args:
            text: The text content of the memory
            title: A title for the memory (auto-generated if not provided)
            source_address: Address of the source (optional)
            metadata: Additional metadata about the memory
            user_id: ID of the user who owns this memory (defaults to default user if not provided)

        Returns:
            The created Memory object
        """
        return self.memory_manager.create_memory(
            text, title, source_address, metadata, user_id
        )

    def add_memory(self, memory: Memory) -> Memory:
        """
        Add an existing memory to storage.

        Args:
            memory: The Memory object to store

        Returns:
            The stored Memory object
        """
        return self.memory_manager.add_memory(memory)

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search for memories relevant to a query.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of relevant Memory objects
        """
        return self.memory_manager.search(query, limit)

    def find_similar_memories(
        self,
        threshold: float = 0.85,
        min_cluster_size: int = 2,
        max_cluster_size: int = 5,
    ) -> List[List[Memory]]:
        """
        Find clusters of similar memories.

        Args:
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of memory clusters, where each cluster is a list of Memory objects
        """
        return self.memory_manager.find_similar_memories(
            threshold, min_cluster_size, max_cluster_size
        )

    def consolidate_memories(
        self,
        model: str,
        threshold: float = 0.85,
        min_cluster_size: int = 2,
        max_cluster_size: int = 5,
    ) -> List[Memory]:
        """
        Consolidate similar memories to create higher-level memories.

        Args:
            model: The model to use for generating consolidated memories
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of newly created consolidated memories
        """
        return self.memory_manager.consolidate_memories(
            model, threshold, min_cluster_size, max_cluster_size
        )

    def get_relevant_memories(self, request: Dict[str, Any]) -> List[Memory]:
        """
        Retrieve relevant memories based on the request.

        Args:
            request: The chat completion request

        Returns:
            List of relevant Memory objects
        """
        return self.memory_manager.get_relevant_memories(request)

    # Chat-related methods delegated to ChatManager
    def augment_chat_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat completion request with memory augmentation.

        Args:
            request: The chat completion request

        Returns:
            Augmented chat completion request
        """
        return self.chat_manager.augment_chat_request(request)

    def completion(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate an OpenAI compatible chat completion with memory augmentation.

        Args:
            **kwargs: Arguments to pass to the chat completion API
                user: Optional[str] - A unique identifier representing the end-user

        Returns:
            Chat completion response
        """
        return self.chat_manager.completion(**kwargs)

    # Message-related methods delegated to MessageManager
    def create_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """
        Create a new message.

        Args:
            role: The role of the message sender (system, user, assistant, tool)
            content: The content of the message
            metadata: Additional metadata about the message
            user_id: ID of the user who owns this message (defaults to default user if not provided)
            tool_call_id: ID of the tool call this message is responding to (for tool messages)
            tool_calls: Tool calls made by this message (for assistant messages)

        Returns:
            The created Message object
        """
        return self.message_manager.create_message(
            role, content, metadata, user_id, tool_call_id, tool_calls
        )

    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            The Message object if found, None otherwise
        """
        return self.message_manager.get_message(message_id)

    def get_all_messages(self) -> List[Message]:
        """
        Get all messages.

        Returns:
            List of all Message objects
        """
        return self.message_manager.get_all_messages()

    def create_message_set(
        self,
        message_ids: List[str],
        active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> MessageSet:
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
        return self.message_manager.create_message_set(
            message_ids, active, metadata, user_id
        )

    def get_message_set(self, message_set_id: str) -> Optional[MessageSet]:
        """
        Get a message set by ID.

        Args:
            message_set_id: The ID of the message set to retrieve

        Returns:
            The MessageSet object if found, None otherwise
        """
        return self.message_manager.get_message_set(message_set_id)

    def get_active_message_set(self) -> Optional[MessageSet]:
        """
        Get the active message set.

        Returns:
            The active MessageSet object if found, None otherwise
        """
        return self.message_manager.get_active_message_set()

    def get_messages_in_set(self, message_set_id: str) -> List[Message]:
        """
        Get all messages in a message set.

        Args:
            message_set_id: The ID of the message set

        Returns:
            List of Message objects in the message set
        """
        return self.message_manager.get_messages_in_set(message_set_id)

    def deactivate_all_message_sets(self) -> None:
        """
        Deactivate all message sets.
        """
        self.message_manager.deactivate_all_message_sets()

    def store_conversation(
        self,
        messages: List[Dict[str, str]],
        response: Any,
        user_id: Optional[int] = None,
    ) -> MessageSet:
        """
        Store a conversation as a message set.

        Args:
            messages: List of messages in the conversation
            response: The response from the LLM
            user_id: ID of the user who owns this conversation

        Returns:
            The created MessageSet object
        """
        return self.message_manager.store_conversation(messages, response, user_id)

    def compress_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        target_token_count: int = 4000,
        max_message_age: Optional[datetime.timedelta] = None,
    ) -> List[Dict[str, str]]:
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
        return self.message_manager.compress_messages(
            model, messages, target_token_count, max_message_age
        )

    # Internal methods that need to remain in the RecallKit class
    class MemoryResponse(BaseModel):
        """Response format for memory consolidation."""

        text: str = Field(..., description="The text content of the memory")
        title: str = Field(
            ..., description="A title or brief description of the memory"
        )

    def _generate_consolidated_memory(
        self, model: str, memories: List[Memory]
    ) -> MemoryResponse:
        """
        Generate text and title for a consolidated memory using LLM.

        Args:
            model: The model to use for generating the consolidated memory
            memories: List of memories to consolidate

        Returns:
            MemoryResponse with text and title for the consolidated memory
        """
        # Prepare the memories as context
        memory_texts = [
            f"Memory {i+1}: {memory.text}" for i, memory in enumerate(memories)
        ]
        memory_context = "\n".join(memory_texts)

        # Create the prompt
        prompt = f"""
        You are tasked with consolidating multiple related memories into a single coherent memory.

        Here are the memories to consolidate:
        {memory_context}

        Please create a consolidated memory that captures the key information from all these memories.
        Provide both a concise title and a comprehensive text that summarizes the information.
        """

        # Call the LLM to generate the consolidated memory
        messages = [{ROLE: USER, CONTENT: prompt}]

        try:
            # Try with response_format parameter (for OpenAI-compatible APIs)
            response = self.completion_fn(
                model=model,
                messages=messages,
                response_format=self.MemoryResponse,
            )
        except Exception:
            # If response_format fails, try without it
            response = self.completion_fn(
                model=model,
                messages=[
                    {
                        ROLE: USER,
                        CONTENT: prompt
                        + "\n\nRespond with a JSON object containing 'text' and 'title' fields.",
                    }
                ],
            )

        # Extract the response content
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"][CONTENT]
        else:
            raise ValueError("Invalid response format from completion function")

        # Parse the JSON response
        memory_data = json.loads(content)
        return self.MemoryResponse(**memory_data)
