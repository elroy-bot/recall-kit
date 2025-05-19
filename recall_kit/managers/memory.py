"""
Memory management functionality for Recall Kit.

This module contains the MemoryManager class for creating, retrieving, and managing memories.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from recall_kit.models import Memory
from recall_kit.utils.embedding import calculate_similarity


class MemoryManager:
    """
    Manager class for memory-related operations.

    This class provides methods for creating, storing, retrieving, and
    consolidating memories.
    """

    def __init__(self, recall_kit):
        """
        Initialize a new MemoryManager instance.

        Args:
            recall_kit: The RecallKit instance this manager belongs to
        """
        self.recall_kit = recall_kit
        self.storage = recall_kit.storage
        self.embedding_fn = recall_kit.embedding_fn

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
        # Get all active memories
        memories = [m for m in self.storage.get_all_memories() if m.active]

        # Group by similarity
        clusters = self._cluster_memories(
            memories, threshold, min_cluster_size, max_cluster_size
        )

        return clusters

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
        # Find clusters of similar memories
        clusters = self.find_similar_memories(
            threshold, min_cluster_size, max_cluster_size
        )

        # Create consolidated memories
        consolidated_memories = []
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            # Create a consolidated memory
            parent_ids = [memory.id for memory in cluster]

            user_id = cluster[0].user_id

            assert all(
                memory.user_id == user_id for memory in cluster
            ), "All memories must belong to the same user"

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

    def _cluster_memories(
        self,
        memories: List[Memory],
        threshold: float,
        min_cluster_size: int,
        max_cluster_size: int,
    ) -> List[List[Memory]]:
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
                similarity = self._calculate_similarity(
                    memory_i.embedding, memory_j.embedding
                )

                if similarity >= threshold and len(cluster) < max_cluster_size:
                    cluster.append(memory_j)
                    used_memories.add(memory_j.id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        return calculate_similarity(embedding1, embedding2)

    def _generate_consolidated_memory(self, model: str, memories: List[Memory]) -> Any:
        """
        Generate text and title for a consolidated memory using LLM.

        Args:
            model: The model to use for generating the consolidated memory
            memories: List of memories to consolidate

        Returns:
            MemoryResponse with text and title for the consolidated memory
        """
        return self.recall_kit._generate_consolidated_memory(model, memories)

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
        memories = self.recall_kit.retrieve_fn(self.storage, self.embedding_fn, request)

        # Filter memories
        if self.recall_kit.filter_fn:
            memories = [m for m in memories if self.recall_kit.filter_fn(m, request)]

        if self.recall_kit.rerank_fn:
            memories = self.recall_kit.rerank_fn(memories, request)

        return memories
