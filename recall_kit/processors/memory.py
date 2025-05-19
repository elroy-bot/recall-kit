import json
from dataclasses import dataclass
from functools import cached_property, partial
from typing import List

import numpy as np
from litellm import AllMessageValues
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from toolz import concat, pipe, unique
from toolz.curried import map, take

from recall_kit.models import Memory

from ..constants import CONTENT, ROLE, USER
from ..protocols.base import (
    CompletionFunction,
    EmbeddingFunction,
    StorageBackendProtocol,
)
from ..utils.completion import extract_content_from_response
from ..utils.messaging import to_user_message


class MemoryResponse(BaseModel):
    text: str = Field(..., description="The text content of the memory")
    title: str = Field(..., description="A title or brief description of the memory")


@dataclass
class MemoryCluster:
    memories: List[Memory]
    embeddings: np.ndarray

    def __len__(self):
        return len(self.memories)

    def __str__(self) -> str:
        # Return a string representation of the object
        return pipe(
            self.memories,
            map(lambda m: m.to_text()),
            list,
            "\n".join,
            lambda x: "#Memory Cluster:\n" + x,
        )  # type: ignore

    def __lt__(self, other: "MemoryCluster") -> bool:
        """Define default sorting behavior.
        First sort by cluster size (larger clusters first)
        Then by mean distance (tighter clusters first)"""

        return self._sort_key < other._sort_key

    @property
    def _sort_key(self):
        # Sort such that clusters early in a list are those that are most in need of consolidation.
        # Sort by: cluster size and then mean distance (ie tightness of cluster)
        return (-len(self), self.mean_distance)

    def token_count(self, chat_model_name: str):
        from litellm.utils import token_counter

        return token_counter(chat_model_name, text=str(self))

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        """Lazily compute and cache the distance matrix."""
        size = len(self)
        _distance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                dist = cosine(self.embeddings[i], self.embeddings[j])
                _distance_matrix[i, j] = dist
                _distance_matrix[j, i] = dist
        return _distance_matrix

    @cached_property
    def mean_distance(self) -> float:
        """Calculate the mean intra cluster distance between all pairs of embeddings in the cluster using cosine similarity"""
        if len(self) < 2:
            return 0.0

        dist_matrix = self.distance_matrix
        # Get upper triangle of matrix (excluding diagonal of zeros)
        upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        return float(np.mean(upper_triangle))

    def get_densest_n(self, n: int = 2) -> "MemoryCluster":
        """Get a new MemoryCluster containing the n members with lowest mean distance to other cluster members.

        Args:
            n: Number of members to return. Defaults to 2.

        Returns:
            A new MemoryCluster containing the n members with lowest mean distance to other members.
        """
        if len(self) <= n:
            return self

        dist_matrix = self.distance_matrix
        # Calculate mean distance for each member (excluding self-distance on diagonal)
        mean_distances = []
        for i in range(len(self)):
            # Get all distances except the diagonal (which is 0)
            member_distances = np.concatenate(
                [dist_matrix[i, :i], dist_matrix[i, i + 1 :]]
            )
            mean_dist = np.mean(member_distances)
            mean_distances.append((mean_dist, i))

        # Sort by mean distance and take top n indices
        mean_distances.sort(key=lambda x: x[0])
        closest_indices = [idx for _, idx in mean_distances[:n]]

        # Create new cluster with selected memories and embeddings
        return MemoryCluster(
            memories=[self.memories[i] for i in closest_indices],
            embeddings=self.embeddings[closest_indices],
        )


class MemoryConsolidator:
    def __init__(
        self,
        embedding_model: str,
        storage: StorageBackendProtocol,
        completion_fn: CompletionFunction,
        embedding_fn: EmbeddingFunction,
        eps: float = 0.85,
        min_samples: int = 2,
        max_samples: int = 5,
    ):
        """
        Initialize the MemoryConsolidator.

        Args:
            storage: The storage backend to use
            completion_fn: The function to call for generating text
            embedding_fn: The function to call for generating embeddings
        """
        self.embedding_model = embedding_model
        self.storage = storage
        self.completion = completion_fn
        self.embedding = embedding_fn
        self.eps = eps
        self.min_samples = min_samples
        self.max_samples = max_samples

    def consolidate_memories(
        self,
        completion_model: str,
        threshold: float = 0.21125,
        min_cluster_size: int = 3,
        max_cluster_size: int = 5,
    ) -> List[Memory]:
        """
        Consolidate similar memories to create higher-level memories.

        Args:
            completion_model: The model to use for generating consolidated memories
            threshold: Similarity threshold for clustering (0-1)
            min_cluster_size: Minimum number of memories to form a cluster
            max_cluster_size: Maximum number of memories to include in a cluster

        Returns:
            List of newly created consolidated memories
        """

        clusters: List[MemoryCluster] = pipe(
            self.storage.get_active_memories(),
            partial(
                self.find_clusters,
                eps=threshold,
                min_samples=min_cluster_size,
                max_samples=max_cluster_size,
            ),
            take(3),
            list,
        )  # type: ignore

        consolidated_memories = []

        for cluster in clusters:
            parent_ids = [memory.id for memory in cluster.memories]

            user_id = cluster.memories[0].user_id

            assert all(
                memory.user_id == user_id for memory in cluster.memories
            ), "All memories must belong to the same user"

            # Generate consolidated memory content using LLM
            memory_response = self.generate_consolidated_memory(
                completion_model, cluster.memories
            )
            source_metadata = pipe(
                cluster.memories,
                map(lambda m: m.source_metadata),
                concat,
                unique,
            )

            # Create the consolidated memory
            consolidated_memory = Memory(
                content=memory_response.text,
                title=memory_response.title,
                _parent_ids=json.dumps(parent_ids),
                user_id=user_id,
                _source_metadata=json.dumps(source_metadata),
            )

            # Store the consolidated memory
            self.storage.store_memory(consolidated_memory)
            consolidated_memories.append(consolidated_memory)

            # Mark original memories as inactive
            for memory in cluster.memories:
                memory.active = False
                self.storage.update_memory(memory)

        return consolidated_memories

    def find_clusters(
        self,
        memories: List[Memory],
        eps: float = 0.21125,
        max_samples: int = 5,
        min_samples: int = 3,
    ) -> List[MemoryCluster]:
        if not memories:
            return []

        embeddings = []
        valid_memories = []
        for memory in memories:
            memory_id = memory.id
            assert memory_id
            embedding = self.storage.fetch_embedding(
                model=self.embedding_model,
                source_type=memory.source_type,
                source_id=memory_id,
            )
            if embedding is not None:
                embeddings.append(embedding)
                valid_memories.append(memory)

        if not embeddings:
            raise ValueError("No embeddings found for memories")

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=eps,
            metric="cosine",
            min_samples=min_samples,
        ).fit(embeddings_array)

        # Group memories by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Skip noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Create MemoryCluster objects
        clusters = pipe(
            [
                MemoryCluster(
                    embeddings=embeddings_array[indices],
                    memories=[valid_memories[i] for i in indices],
                )
                for indices in clusters.values()
            ],
            map(lambda x: x.get_densest_n(max_samples)),
            list,
            partial(sorted),
        )

        return clusters

    def generate_consolidated_memory(
        self, completion_model: str, memories: List[Memory]
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
            f"Memory {i+1}: {memory.content}" for i, memory in enumerate(memories)
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
        messages: List[AllMessageValues] = [to_user_message(content=prompt)]

        try:
            # Try with response_format parameter (for OpenAI-compatible APIs)
            response = self.completion(
                model=completion_model,
                messages=messages,
                response_format=MemoryResponse,  # type: ignore
            )

        except Exception:
            # If response_format fails, try without it
            response = self.completion(
                model=completion_model,
                messages=[
                    {
                        ROLE: USER,
                        CONTENT: prompt
                        + "\n\nRespond with a JSON object containing 'text' and 'title' fields.",
                    }
                ],
            )

        # Extract the response content
        content = extract_content_from_response(response)

        # Parse the JSON response
        assert isinstance(content, str), "Response content should be a string"
        memory_data = json.loads(content)
        return MemoryResponse(**memory_data)
