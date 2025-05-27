import json
from typing import List, Optional

from toolz import pipe

from recall_kit.models import Memory

from ..models.pydantic_models import SourceMetadata
from ..protocols.base import (
    EmbeddingFunction,
    FilterFunction,
    RerankFunction,
    StorageBackendProtocol,
)


class MemoryService:
    def __init__(
        self,
        storage: StorageBackendProtocol,
        embedding_model: str,
        embedding_fn: EmbeddingFunction,
        filter_fn: FilterFunction,
        rerank_fn: RerankFunction,
    ):
        """
        Initialize a new MemoryStore instance.

        Args:
            storage: The storage backend to use for memory management
        """
        self.storage = storage
        self.embedding_fn = embedding_fn
        self.embedding_model = embedding_model
        self.filter_fn = filter_fn
        self.rerank_fn = rerank_fn

    def create_memory(
        self,
        text: str,
        title: Optional[str] = None,
        source_address: Optional[str] = None,
        source_metadata: List[SourceMetadata] = [],
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

        return pipe(
            self.embedding_fn(self.embedding_model, query),
            lambda e: self.storage.search_memories(e, limit=limit),
            lambda memories: self.filter_fn(None, memories),
            lambda memories: self.rerank_fn(None, memories),
            list,
        )  # type: ignore

    def get_all_memories(self) -> List[Memory]:
        """
        Get all memories.

        Returns:
            List of all Memory objects
        """
        return self.storage.get_active_memories()
