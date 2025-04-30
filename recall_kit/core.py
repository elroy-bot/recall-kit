"""
Core data structures and functionality for Recall Kit.

This module contains the main classes and functions for working with memories,
including the RecallKit class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeVar, Protocol

import numpy as np
from pydantic import BaseModel, Field

from .constants import MESSAGES
from recall_kit.models import Memory, Message, MessageSet
from litellm import ModelResponse, Type

from typing import Protocol, runtime_checkable


# Define type protocols for the callback functions
@runtime_checkable
class RetrieveFunction(Protocol):
    def __call__(self, storage: StorageBackendProtocol, embedding_fn: EmbeddingFunction, request: Any) -> List[Memory]: ...

@runtime_checkable
class FilterFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> bool: ...

@runtime_checkable
class RerankFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> List[Memory]: ...

@runtime_checkable
class AugmentFunction(Protocol):
    def __call__(self, memories: List[Memory], request: Any) -> Any: ...

@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, text: str) -> List[float]: ...

@runtime_checkable
class CompletionFunction(Protocol):
    def __call__(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ) -> ModelResponse: ...

@runtime_checkable
class StorageBackendProtocol(Protocol):
    def store_memory(self, memory: Any) -> None: ...
    def get_memory(self, memory_id: str) -> Optional[Any]: ...
    def get_all_memories(self) -> List[Any]: ...
    def search_memories(self, query_embedding: List[float], limit: int = 5) -> List[Any]: ...
    def update_memory(self, memory: Any) -> None: ...
    def delete_memory(self, memory_id: str) -> bool: ...
    def store_message(self, message: Any) -> None: ...
    def get_message(self, message_id: str) -> Optional[Any]: ...
    def get_all_messages(self) -> List[Any]: ...
    def store_message_set(self, message_set: Any) -> None: ...
    def get_message_set(self, message_set_id: str) -> Optional[Any]: ...
    def get_active_message_set(self) -> Optional[Any]: ...
    def get_messages_in_set(self, message_set_id: str) -> List[Any]: ...
    def deactivate_all_message_sets(self) -> None: ...
    def get_all_message_sets(self) -> List[Any]: ...
    def create_user(self, token: str) -> int: ...
    def get_user_by_token(self, token: str) -> Optional[int]: ...
    def get_default_user_id(self) -> int: ...

# Type variable for the RecallKit class
T = TypeVar('T', bound='RecallKit')


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
            get_storage_backend, get_embedding_fn, get_completion_fn,
            get_retrieve_fn, get_filter_fn, get_rerank_fn, get_augment_fn
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

    def create_memory(self, text: str, title: Optional[str] = None, source_address: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None, user_id: Optional[int] = None) -> Memory:
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
        if not title:
            # Generate a title if not provided
            title = text[:50] + "..." if len(text) > 50 else text

        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        memory = Memory(
            text=text,
            title=title,
            source_address=source_address,
            metadata=metadata or {},
            user_id=user_id,
        )

        # Generate embedding
        memory.embedding = self.embedding_fn(text)

        # Store the memory
        self.storage.store_memory(memory)

        return memory

    def add_memory(self, memory: Memory) -> Memory:
        """
        Add an existing memory to storage.

        Args:
            memory: The Memory object to store

        Returns:
            The stored Memory object
        """
        if memory.embedding is None:
            memory.embedding = self.embedding_fn(memory.text)

        self.storage.store_memory(memory)
        return memory

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search for memories relevant to a query.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of relevant Memory objects
        """
        query_embedding = self.embedding_fn(query)
        return self.storage.search_memories(query_embedding, limit=limit)

    def find_similar_memories(self, threshold: float = 0.85, min_cluster_size: int = 2,
                             max_cluster_size: int = 5) -> List[List[Memory]]:
        """
        Find clusters of similar memories.

        Args:
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of memory clusters, where each cluster is a list of Memory objects
        """
        # Get all active memories
        memories = [m for m in self.storage.get_all_memories() if m.active]

        # Group by similarity
        clusters = self._cluster_memories(memories, threshold, min_cluster_size, max_cluster_size)

        return clusters

    def consolidate_memories(self, model: str, threshold: float = 0.85, min_cluster_size: int = 2,
                            max_cluster_size: int = 5) -> List[Memory]:
        """
        Consolidate similar memories to create higher-level memories.

        Args:
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of newly created consolidated memories
        """
        # Find clusters of similar memories
        clusters = self.find_similar_memories(threshold, min_cluster_size, max_cluster_size)

        # Create consolidated memories
        consolidated_memories = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            # Create a consolidated memory
            parent_ids = [memory.id for memory in cluster]

            user_id = cluster[0].user_id

            assert all(memory.user_id == user_id for memory in cluster), "All memories must belong to the same user"

            # Generate consolidated memory content using LLM
            memory_response = self._generate_consolidated_memory(model, cluster)

            consolidated_memory = Memory(
                text=memory_response.text,
                title=memory_response.title,
                parent_ids=parent_ids,
                metadata={"consolidated": True, "parent_count": len(parent_ids)},
                user_id=user_id,
            )

            # Generate embedding
            consolidated_memory.embedding = self.embedding_fn(memory_response.text)

            # Store the consolidated memory
            self.storage.store_memory(consolidated_memory)

            # Mark original memories as inactive
            for memory in cluster:
                memory.active = False
                self.storage.update_memory(memory)

            consolidated_memories.append(consolidated_memory)

        return consolidated_memories

    def _cluster_memories(self, memories: List[Memory], threshold: float,
                         min_cluster_size: int, max_cluster_size: int) -> List[List[Memory]]:
        """
        Cluster memories by similarity.

        Args:
            memories: List of memories to cluster
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of memory clusters, where each cluster is a list of Memory objects
        """
        # Skip if there are too few memories
        if len(memories) < min_cluster_size:
            return []

        # Simple clustering algorithm
        clusters = []
        used_memories = set()

        for i, memory_i in enumerate(memories):
            if memory_i.id in used_memories:
                continue

            cluster = [memory_i]
            used_memories.add(memory_i.id)

            for j, memory_j in enumerate(memories):
                if i == j or memory_j.id in used_memories:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(memory_i.embedding, memory_j.embedding)

                if similarity >= threshold and len(cluster) < max_cluster_size:
                    cluster.append(memory_j)
                    used_memories.add(memory_j.id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    class MemoryResponse(BaseModel):
        """Response format for memory consolidation."""
        text: str = Field(..., description="The text content of the memory")
        title: str = Field(..., description="A title or brief description of the memory")

    def _generate_consolidated_memory(self, model: str, memories: List[Memory]) -> MemoryResponse:
        """
        Generate text and title for a consolidated memory using LLM.

        Args:
            memories: List of memories to consolidate

        Returns:
            MemoryResponse with text and title for the consolidated memory
        """
        # Prepare the memories as context
        memory_texts = [f"Memory {i+1}: {memory.text}" for i, memory in enumerate(memories)]
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
        messages = [{"role": "user", "content": prompt}]
        # Try with response_format parameter (for OpenAI-compatible APIs)
        response = self.completion_fn(
            model=model,
            messages=messages,
            additional_args={
                "response_format": {"type": "json_object", "schema": self.MemoryResponse.model_json_schema()}
            }
        )


        # Extract the response content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response:
            content = response['choices'][0]['message']['content']
            # Parse the JSON response
        import json
        memory_data = json.loads(content)
        return self.MemoryResponse(**memory_data)


    def get_relevant_memories(self, request: Dict[str, Any]) -> List[Memory]:
        """
        Retrieve relevant memories based on the request.

        Args:
            request: The chat completion request

        Returns:
            List of relevant Memory objects
        """
        # Extract the query from the last user message
        messages = request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]

        if not user_messages:
            return []

        # Retrieve relevant memories
        memories = self.retrieve_fn(self.storage, self.embedding_fn, request)

        # Filter memories
        if self.filter_fn:
            memories = [m for m in memories if self.filter_fn(m, request)]

        if self.rerank_fn:
            memories = self.rerank_fn(memories, request)

        return memories


    def augment_chat_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat completion request with memory augmentation.

        Args:
            request: The chat completion request

        Returns:
            Augmented chat completion request
        """
        # Extract the query from the last user message
        memories = self.get_relevant_memories(request)
        # Augment the request with memories
        augmented_request = self.augment_fn(memories, request)

        return augmented_request

    def compress_messages(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo",
                         target_token_count: int = 4000, max_message_age: Optional[datetime.timedelta] = None) -> List[Dict[str, str]]:
        """
        Compress messages to fit within the context window, by summarizing earlier messages.

        Creates memories from compressed messages that are dropped from the context.

        Args:
            messages: List of messages to compress
            model: Model name to use for token counting
            target_token_count: Target number of tokens to keep
            max_message_age: Maximum age of messages to keep (None means no limit)

        Returns:
           Compressed messages
        """
        import datetime
        from litellm import token_counter
        from collections import deque

        if not messages:
            return []

        # Find system message if it exists
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        # If no system message, we'll just work with all messages
        if system_messages:
            system_message = system_messages[0]
            current_token_count = token_counter(model=model, text=system_message.get("content", ""))
        else:
            system_message = None
            current_token_count = 0

        kept_messages = deque()
        dropped_messages = []

        # Process messages in reverse order (newest first)
        for msg in reversed(non_system_messages):
            msg_content = msg.get("content", "")
            msg_role = msg.get("role", "")

            # Calculate tokens for this message
            msg_token_count = token_counter(model=model, text=msg_content)

            # Check if we need to keep this message due to tool calls
            if len(kept_messages) > 0 and kept_messages[0].get("role") == "tool" and msg_role == "assistant":
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
            if max_message_age and "created_at" in msg:
                msg_created_at = msg.get("created_at")
                if isinstance(msg_created_at, str):
                    msg_created_at = datetime.datetime.fromisoformat(msg_created_at)

                if msg_created_at < datetime.datetime.now() - max_message_age:
                    dropped_messages.append(msg)
                    continue

            # If we get here, keep the message
            kept_messages.appendleft(msg)
            current_token_count += msg_token_count

        # Create memories from dropped messages
        if dropped_messages:
            # Create a consolidated memory from the dropped messages
            dropped_text = "\n".join([
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in dropped_messages
            ])

            # Create a memory from the dropped messages
            self.create_memory(
                text=dropped_text,
                title=f"Compressed messages from conversation",
                metadata={"compressed": True, "message_count": len(dropped_messages)},
            )

        # Construct the final message list
        result = list(kept_messages)
        if system_message:
            result.insert(0, system_message)

        return result

    def completion(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate an OpenAI compatible chat completion with memory augmentation.

        Args:
            **kwargs: Arguments to pass to the chat completion API
                user: Optional[str] - A unique identifier representing the end-user

        Returns:
            Chat completion response
        """
        from litellm import ContextWindowExceededError

        # Process the request with memory augmentation
        augmented_request = self.augment_chat_request(kwargs)

        # Extract parameters for the completion function
        model = augmented_request.pop("model", None)
        if not model:
            raise ValueError("No model specified for chat completion")

        messages = augmented_request.pop(MESSAGES, [])
        max_tokens = augmented_request.pop("max_tokens", None)
        temperature = augmented_request.pop("temperature", None)

        # Extract user parameter if provided
        user_token = augmented_request.pop("user", "default")

        # Get or create user ID from token
        user_id = self.storage.get_user_by_token(user_token)
        if user_id is None:
            user_id = self.storage.create_user(user_token)

        # Add any remaining kwargs to additional args
        additional_args = augmented_request
        # Add user token back to additional args for the completion function
        additional_args["user"] = user_token

        # Generate the completion using the completion function
        try:
            response = self.completion_fn(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_args=additional_args
            )
        except ContextWindowExceededError as e:
            kwargs[MESSAGES]= self.compress_messages(messages)
            return self.completion(kwargs)

        # Store the conversation as a memory and as messages
        self._store_conversation_memory({"messages": messages, "model": model, "user_id": user_id}, response)
        self.store_conversation(messages, response, user_id)

        return response

    def _store_conversation_memory(self, request: Dict[str, Any], response: Any) -> None:
        """
        Store a conversation as a memory.

        Args:
            request: The chat completion request
            response: The chat completion response
        """
        messages = request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        user_id = request.get("user_id")

        if not user_messages:
            return

        query = user_messages[-1].get("content", "")

        try:
            # Handle both object-style and dict-style responses
            if hasattr(response, 'choices'):
                assistant_message = response.choices[0].message.content
            elif isinstance(response, dict) and 'choices' in response:
                choice = response['choices'][0]
                if isinstance(choice, dict) and 'message' in choice:
                    assistant_message = choice['message'].get('content', '')
                else:
                    return
            else:
                return
        except (AttributeError, IndexError, KeyError):
            return

        conversation_text = f"User: {query}\nAssistant: {assistant_message}"

        self.create_memory(
            text=conversation_text,
            title=query[:50] + "..." if len(query) > 50 else query,
            metadata={"type": "conversation"},
            user_id=user_id
        )

    def create_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[int] = None) -> Message:
        """
        Create a new message.

        Args:
            role: The role of the message sender (system, user, assistant)
            content: The content of the message
            metadata: Additional metadata about the message
            user_id: ID of the user who owns this message (defaults to default user if not provided)

        Returns:
            The created Message object
        """
        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            user_id=user_id,
        )

        # Store the message
        self.storage.store_message(message)

        return message

    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            The Message object if found, None otherwise
        """
        return self.storage.get_message(message_id)

    def get_all_messages(self) -> List[Message]:
        """
        Get all messages.

        Returns:
            List of all Message objects
        """
        return self.storage.get_all_messages()

    def create_message_set(self, message_ids: List[str], active: bool = True,
                           metadata: Optional[Dict[str, Any]] = None, user_id: Optional[int] = None) -> MessageSet:
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

        message_set = MessageSet(
            message_ids=message_ids,
            active=active,
            metadata=metadata or {},
            user_id=user_id,
        )

        # Store the message set
        self.storage.store_message_set(message_set)

        return message_set

    def get_message_set(self, message_set_id: str) -> Optional[MessageSet]:
        """
        Get a message set by ID.

        Args:
            message_set_id: The ID of the message set to retrieve

        Returns:
            The MessageSet object if found, None otherwise
        """
        return self.storage.get_message_set(message_set_id)

    def get_active_message_set(self) -> Optional[MessageSet]:
        """
        Get the active message set.

        Returns:
            The active MessageSet object if found, None otherwise
        """
        return self.storage.get_active_message_set()

    def get_messages_in_set(self, message_set_id: str) -> List[Message]:
        """
        Get all messages in a message set.

        Args:
            message_set_id: The ID of the message set

        Returns:
            List of Message objects in the message set
        """
        return self.storage.get_messages_in_set(message_set_id)

    def deactivate_all_message_sets(self) -> None:
        """
        Deactivate all message sets.
        """
        self.storage.deactivate_all_message_sets()

    def store_conversation(self, messages: List[Dict[str, str]], response: Any, user_id: Optional[int] = None) -> MessageSet:
        """
        Store a conversation as a message set.

        Args:
            messages: List of messages in the conversation
            response: The response from the LLM
            user_id: ID of the user who owns this conversation

        Returns:
            The created MessageSet object
        """
        # Get the active message set
        active_message_set = self.get_active_message_set()

        # If there's only one user message and an active message set, add to it
        if len(messages) == 1 and messages[0].get("role") == "user" and active_message_set:
            # Create a new message for the user input
            user_message = self.create_message(
                role="user",
                content=messages[0].get("content", ""),
                metadata={"type": "conversation"},
                user_id=user_id,
            )

            # Create a new message for the assistant response
            try:
                # Handle both object-style and dict-style responses
                if hasattr(response, 'choices'):
                    assistant_content = response.choices[0].message.content
                elif isinstance(response, dict) and 'choices' in response:
                    choice = response['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice:
                        assistant_content = choice['message'].get('content', '')
                    else:
                        return active_message_set
                else:
                    return active_message_set

                assistant_message = self.create_message(
                    role="assistant",
                    content=assistant_content,
                    metadata={"type": "conversation"},
                    user_id=user_id,
                )

                # Update the message set with the new messages
                message_ids = active_message_set.message_ids + [user_message.id, assistant_message.id]

                # Create a new message set with the updated message IDs
                return self.create_message_set(
                    message_ids=message_ids,
                    active=True,
                    metadata={"type": "conversation"},
                    user_id=user_id,
                )
            except (AttributeError, IndexError):
                # If there's an error getting the assistant response, just return the active message set
                return active_message_set
        else:
            # Create new messages for each message in the conversation
            message_ids = []

            # If there's an active message set, check for duplicate messages
            existing_messages = []
            if active_message_set:
                existing_messages = self.get_messages_in_set(active_message_set.id)

            # Process each message
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Check if this message already exists in the active message set
                duplicate = False
                for existing_msg in existing_messages:
                    if existing_msg.role == role and existing_msg.content == content:
                        message_ids.append(existing_msg.id)
                        duplicate = True
                        break

                # If not a duplicate, create a new message
                if not duplicate:
                    message = self.create_message(
                        role=role,
                        content=content,
                        metadata={"type": "conversation"},
                        user_id=user_id,
                    )
                    message_ids.append(message.id)

            # Create a message for the assistant response
            try:
                # Handle both object-style and dict-style responses
                if hasattr(response, 'choices'):
                    assistant_content = response.choices[0].message.content
                elif isinstance(response, dict) and 'choices' in response:
                    choice = response['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice:
                        assistant_content = choice['message'].get('content', '')
                    else:
                        # Skip adding assistant message if we can't extract content
                        return self.create_message_set(
                            message_ids=message_ids,
                            active=True,
                            metadata={"type": "conversation"},
                            user_id=user_id,
                        )
                else:
                    # Skip adding assistant message if response format is unexpected
                    return self.create_message_set(
                        message_ids=message_ids,
                        active=True,
                        metadata={"type": "conversation"},
                        user_id=user_id,
                    )

                assistant_message = self.create_message(
                    role="assistant",
                    content=assistant_content,
                    metadata={"type": "conversation"},
                    user_id=user_id,
                )
                message_ids.append(assistant_message.id)
            except (AttributeError, IndexError, KeyError):
                # If there's an error getting the assistant response, continue without it
                pass

            # Create a new message set with the messages
            return self.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"type": "conversation"},
                user_id=user_id,
            )
