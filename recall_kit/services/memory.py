import json
from typing import Any, Dict, List, Optional

from recall_kit.models import Memory

from ..protocols.base import EmbeddingFunction, StorageBackendProtocol


class MemoryService:
    def __init__(
        self,
        storage: StorageBackendProtocol,
        embedding_model: str,
        embedding_fn: EmbeddingFunction,
    ):
        """
        Initialize a new MemoryStore instance.

        Args:
            storage: The storage backend to use for memory management
        """
        self.storage = storage
        self.embedding_fn = embedding_fn
        self.embedding_model = embedding_model

    def create_memory(
        self,
        text: str,
        title: Optional[str] = None,
        source_address: Optional[str] = None,
        source_metadata: List[Dict[str, Any]] = [],
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
        if not title:
            # Generate a title if not provided
            title = text[:50] + "..." if len(text) > 50 else text

        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        assert isinstance(user_id, int), "user_id must be an integer"

        memory = Memory(
            content=text,
            title=title,
            source_address=source_address,
            _source_metadata=json.dumps(source_metadata or {}),
            user_id=user_id,
        )

        # Store the memory
        self.storage.store_memory(memory)

        return memory

    def search_memories(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search for memories relevant to a query (alias for search).

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of relevant Memory objects
        """

        self.storage.get_active_memories()

        return self.storage.search_memories(
            self.embedding_fn(self.embedding_model, query), limit=limit
        )

    def get_all_memories(self) -> List[Memory]:
        """
        Get all memories.

        Returns:
            List of all Memory objects
        """
        return self.storage.get_active_memories()
