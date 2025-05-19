from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pytest
from litellm import AllMessageValues

from recall_kit.core import RecallKit
from recall_kit.models import Embedding, Memory, Message, MessageSet
from recall_kit.storage.sqlite import SQLiteBackend


@pytest.fixture(scope="function")
def storage() -> Generator[SQLiteBackend, Any, None]:
    """Create a temporary SQLite database for testing."""
    from recall_kit.storage.sqlite import SQLiteBackend

    # Create an in-memory SQLite database
    db = SQLiteBackend(":memory:")

    # Initialize the database
    db._initialize_db()

    yield db

    # Cleanup
    db.close()


@pytest.fixture(scope="function")
def recall_kit(storage):
    yield RecallKit(storage=storage)


# Mock embedding service for testing
def mock_embed_text(model: str, text: str) -> List[float]:
    """Create a deterministic mock embedding based on the text."""
    # Use hash of text to seed the random number generator
    np.random.seed(hash(text) % 2**32)

    # Generate a random vector of length 4 (small for testing)
    vector = np.random.normal(0, 1, 4)

    # Normalize to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


class MockStorageBackend:
    """Mock storage backend for testing."""

    def __init__(self):
        self._memories: Dict[int, Memory] = {}
        self._messages: Dict[int, Message] = {}
        self._message_sets: Dict[int, MessageSet] = {}
        self.users = {"default": 1}  # Default user with ID 1
        self.next_user_id = 2

    def store_memory(self, memory):
        self._memories[memory.id] = memory

    def get_active_memories(self):
        return [memory for memory in self._memories.values() if memory.active]

    def store_embedding(
        self, model: str, source_type: str, source_id: int, embedding: List[float]
    ):
        pass

    def get_memory(self, memory_id):
        return self._memories.get(memory_id)

    def search_memories(self, query_embedding, limit=5):
        # Simple mock implementation that returns all memories
        return list(self._memories.values())[:limit]

    def update_memory(self, memory):
        self._memories[memory.id] = memory

    def delete_memory(self, memory_id):
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def store_message(self, message) -> int:
        self._messages[message.id] = message
        return message.id

    def store_messages(self, messages: List[AllMessageValues]) -> List[int]:
        """Store multiple messages and return their IDs."""
        message_ids = []
        for message in messages:
            message_id = self.store_message(message)
            message_ids.append(message_id)
        return message_ids

    def get_message(self, message_id) -> AllMessageValues:
        return self._messages[message_id].data

    def store_message_set(self, message_set) -> int:
        self._message_sets[message_set.id] = message_set
        return message_set.id

    def get_message_set(self, message_set_id):
        return self._message_sets.get(message_set_id)

    def get_active_message_set(self):
        for message_set in self._message_sets.values():
            if message_set.active:
                return message_set
        return None

    def fetch_embedding(
        self, model: str, source_type: str, source_id: int
    ) -> Optional[Embedding]:
        """Fetch an embedding for a given source type and ID."""
        # Mock implementation: return None
        return None

    def get_messages_in_set(self, message_set_id) -> List[AllMessageValues]:
        message_set = self.get_message_set(message_set_id)
        if not message_set:
            return []
        return [self.get_message(msg_id) for msg_id in message_set.message_ids]

    def deactivate_all_message_sets(self):
        for message_set in self._message_sets.values():
            message_set.active = False

    def create_user(self, token: str) -> int:
        """Create a new user with the given token."""
        if token in self.users:
            return self.users[token]

        user_id = self.next_user_id
        self.users[token] = user_id
        self.next_user_id += 1
        return user_id

    def get_user_by_token(self, token: Optional[str]) -> Optional[int]:
        """Get the user ID for the given token."""
        if token is None:
            return None
        return self.users.get(token)

    def get_default_user_id(self) -> int:
        """Get the default user ID."""
        return self.users["default"]
