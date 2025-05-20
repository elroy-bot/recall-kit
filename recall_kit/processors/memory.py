import json
from typing import List

from pydantic import BaseModel, Field

from ..constants import CONTENT, ROLE, USER
from ..models.memory import Memory
from ..protocols.base import (
    CompletionFunction,
    EmbeddingFunction,
    StorageBackendProtocol,
)
from ..utils.embedding import calculate_similarity


class MemoryResponse(BaseModel):
    text: str = Field(..., description="The text content of the memory")
    title: str = Field(..., description="A title or brief description of the memory")


class MemoryConsolidator:
    def __init__(
        self,
        storage: StorageBackendProtocol,
        completion_fn: CompletionFunction,
        embedding_fn: EmbeddingFunction,
    ):
        """
        Initialize the MemoryConsolidator.

        Args:
            storage: The storage backend to use
            completion_fn: The function to call for generating text
            embedding_fn: The function to call for generating embeddings
        """
        self.storage = storage
        self.completion = completion_fn
        self.embedding = embedding_fn

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
            threshold,
            min_cluster_size,
            max_cluster_size,
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
            memory_response = self.generate_consolidated_memory(model, cluster)

            consolidated_memory = Memory(
                text=memory_response.text,
                title=memory_response.title,
                parent_ids=parent_ids,
                metadata={"consolidated": True, "parent_count": len(parent_ids)},
                embedding=self.embedding(memory_response.text),
                user_id=user_id,
                source_address="TODO_CONSOLIDATED_MEMROY_SOURCE_ADDRESS",
            )

            # Store the consolidated memory
            self.storage.store_memory(consolidated_memory)

            # Mark original memories as inactive
            for memory in cluster:
                memory.active = False
                self.storage.update_memory(memory)

            consolidated_memories.append(consolidated_memory)

        return consolidated_memories

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
        clusters = cluster_memories(
            memories, threshold, min_cluster_size, max_cluster_size
        )

        return clusters

    def generate_consolidated_memory(
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
            response = self.completion(
                model=model,
                messages=messages,
                response_format=MemoryResponse,
            )

        except Exception:
            # If response_format fails, try without it
            response = self.completion(
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
            content = response.choices[0].message.content  # type: ignore
        elif isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"][CONTENT]
        else:
            raise ValueError("Invalid response format from completion function")

        # Parse the JSON response
        assert isinstance(content, str), "Response content should be a string"
        memory_data = json.loads(content)
        return MemoryResponse(**memory_data)


def cluster_memories(
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

            assert memory_i.embedding is not None
            assert memory_j.embedding is not None

            # Calculate similarity
            similarity = calculate_similarity(memory_i.embedding, memory_j.embedding)

            if similarity >= threshold and len(cluster) < max_cluster_size:
                cluster.append(memory_j)
                used_memories.add(memory_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters
