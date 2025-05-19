from typing import List

from recall_kit.models import Recallable

from ..protocols.base import EmbeddingFunction, StorageBackendProtocol
from ..utils.embedding import bytes_to_embedding, truncate_if_context_exceeded


class EmbeddingService:
    def __init__(
        self,
        storage: StorageBackendProtocol,
        embedding_fn: EmbeddingFunction,
        model: str,
    ):
        """
        Initialize a new EmbeddingService instance.

        Args:
            storage: The storage backend to use for memory management
            embedding_fn: The function to call for generating embeddings
        """
        self.storage = storage
        self.embedding = truncate_if_context_exceeded(embedding_fn)
        self.model = model

    def upsert_embedding(self, recallable: Recallable) -> List[float]:
        """
        Upsert the embedding for a memory.

        Args:
            memory: The Memory object to upsert
        """
        assert recallable.id

        existing_embedding = self.storage.fetch_embedding(
            self.model, recallable.source_type, recallable.id
        )

        if existing_embedding and existing_embedding.md5 == recallable.md5:
            # No changes to the embedding, no need to update
            return bytes_to_embedding(existing_embedding.embedding)

        else:
            calculated_embedding = self.embedding(self.model, recallable.to_text())
            self.storage.store_embedding(
                self.model, recallable.source_type, recallable.id, calculated_embedding
            )
            return calculated_embedding
