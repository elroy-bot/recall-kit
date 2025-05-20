import logging
from typing import List

from litellm.exceptions import ContextWindowExceededError

from ..protocols.base import EmbeddingFunction, StorageBackendProtocol
from ..storage.base import Recallable
from ..utils.embedding import bytes_to_embedding


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
        self.embedding = embedding_fn
        self.model = model

    def calculate_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using litellm.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            List of float values representing the embedding
        """
        assert isinstance(text, str), "Text must be a string"

        try:
            return self.embedding(
                model=self.model,
                text=text,
            ).data[  # type: ignore
                0
            ]["embedding"]
        except ContextWindowExceededError:
            logging.info("Context window exceeded, retrying with half the text")
            return self.calculate_embedding(text[int(len(text) / 2) :])

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
            calculated_embedding = self.calculate_embedding(recallable.to_text())
            self.storage.store_embedding(
                self.model, recallable.source_type, recallable.id, calculated_embedding
            )
            return calculated_embedding
