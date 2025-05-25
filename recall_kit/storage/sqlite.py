"""
SQLite storage backend for Recall Kit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from typing import List, Optional, Tuple

from litellm import AllMessageValues
from sqlmodel import Session, SQLModel, create_engine, desc, select

from recall_kit.models import Embedding, Memory, Message, MessageSet, User
from recall_kit.protocols import StorageBackendProtocol

from ..utils.embedding import embedding_to_bytes

# Set up logging
logger = logging.getLogger(__name__)


class SQLiteBackend:
    """SQLite storage backend"""

    __protocol_class__ = StorageBackendProtocol

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the SQLite backend.

        Args:
            connection_string: Path to the SQLite database file. If None, uses a default path.
        """
        if connection_string is None:
            # Use a default path in the user's home directory
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, ".recall-kit")
            os.makedirs(db_dir, exist_ok=True)
            connection_string = os.path.join(db_dir, "memories.db")

        self.connection_string = connection_string
        self.engine = create_engine(f"sqlite:///{connection_string}")
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        conn = sqlite3.connect(self.connection_string)
        conn.row_factory = sqlite3.Row

        # Enable loading extensions
        conn.enable_load_extension(True)

        return conn

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        # Create SQLModel tables
        SQLModel.metadata.create_all(self.engine)

        # Create default user if it doesn't exist
        self._create_default_user()

        # Custom setup for vector table
        conn = self._get_connection()
        cursor = conn.cursor()

        # Import sqlite_vec module if available
        try:
            import sqlite_vec

            logger.debug("Attempting to load sqlite_vec extension...")

            # Enable extension loading
            conn.enable_load_extension(True)

            # Load the extension using the sqlite_vec module
            sqlite_vec.load(conn)

            # Disable extension loading after we're done
            conn.enable_load_extension(False)

            logger.debug("Successfully loaded sqlite_vec extension")
        except ImportError:
            # If sqlite_vec module is not available, try direct loading
            logger.debug("sqlite_vec module not found, trying direct loading...")
            cursor.execute("SELECT load_extension('sqlite-vec')")
            logger.debug("Successfully loaded sqlite-vec extension directly")

        # Create vector index if sqlite-vec is available
        try:
            cursor.execute(
                """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_index USING vss0(
                embedding(1536),
                id UNINDEXED,
                text UNINDEXED
            )
            """
            )
        except sqlite3.OperationalError:
            # Create a regular table for testing if the extension is not available
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS memories_index (
                rowid INTEGER PRIMARY KEY,
                embedding TEXT,
                id TEXT,
                text TEXT
            )
            """
            )

        conn.commit()
        conn.close()

        # Check if we need to migrate from old schema to new schema
        # This checks if the memories table has an embedding column
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(memories)")
            conn.close()

        except Exception as e:
            logger.error(f"Error checking for migration need: {e}")

    def get_active_memories(self) -> List[Memory]:
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(Memory)
                    .where(Memory.active == True)
                    .order_by(desc(Memory.created_at))
                ).all()
            )

    def store_embedding(
        self, model: str, source_type: str, source_id: int, embedding: List[float]
    ) -> None:
        embedding_bytes = embedding_to_bytes(embedding)

        with Session(self.engine) as session:
            # Check if we already have an embedding for this memory
            statement = select(Embedding).where(
                Embedding.source_type == source_type,
                Embedding.source_id == source_id,
                Embedding.model_name == model,
            )
            existing_embedding = session.exec(statement).first()

            if existing_embedding:
                # Update existing embedding
                existing_embedding.embedding = embedding_bytes
                existing_embedding = session.merge(existing_embedding)
            else:
                # Create new embedding record
                embedding_table = Embedding(
                    model_name=model,
                    source_type=source_type,
                    source_id=source_id,
                    embedding=embedding_bytes,
                    md5=hashlib.md5(embedding_bytes).hexdigest(),
                )
                session.add(embedding_table)

            session.commit()

    def store_memory(self, memory: Memory) -> None:
        """
        Store a memory in the SQLite database.

        Args:
            memory: The Memory object to store
        """
        # Ensure user_id is set
        if not hasattr(memory, "user_id") or memory.user_id is None:
            memory.user_id = self.get_default_user_id()

        # Create SQLModel object for memory
        stored_memory = Memory(
            id=memory.id,
            content=memory.content,
            title=memory.title,
            source_address=memory.source_address,
            _parent_ids=memory._parent_ids,
            created_at=memory.created_at,
            _source_metadata=memory._source_metadata,
            active=memory.active,
            user_id=memory.user_id,
        )

        # Insert or update memory in database
        with Session(self.engine) as session:
            # Merge will insert or update as needed
            session.merge(stored_memory)
            session.commit()

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            The Memory object if found, None otherwise
        """
        with Session(self.engine) as session:
            return session.get(Memory, memory_id)

    def get_memories_with_embeddings(self) -> List[Tuple[Memory, Embedding]]:
        """
        Retrieve all active memories with their corresponding embeddings.

        Returns:
            List of tuples containing (Memory, Embedding) pairs
        """
        # Get all active memories
        active_memories = self.get_active_memories()

        if not active_memories:
            return []

        # Extract memory IDs for efficient querying
        memory_ids = [memory.id for memory in active_memories]

        # Create a dictionary to map memory IDs to memory objects for quick lookup
        memory_dict = {memory.id: memory for memory in active_memories}

        result = []  # Initialize result outside the session block

        # Fetch all embeddings for these memories in a single query
        with Session(self.engine) as session:
            # Get all embeddings for Memory type
            embeddings = session.exec(
                select(Embedding).where(
                    Embedding.source_type == Memory.__name__,
                    (Embedding.source_id.in_([int(id) for id in memory_ids])),  # type: ignore
                )
            )

            # Pair each memory with its corresponding embedding
            for embedding in embeddings:
                memory = memory_dict.get(embedding.source_id)
                if memory:
                    result.append((memory, embedding))

        return result  # Return result outside the session block

    def search_memories(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Memory]:
        """
        Search for memories similar to the query embedding.

        Args:
            query_embedding: The embedding vector to search for
            limit: Maximum number of results to return

        Returns:
            List of Memory objects, sorted by relevance
        """
        RELEVANCE_THRESHOLD = 0.7

        from ..utils.embedding import bytes_to_embedding, calculate_similarity

        memories_with_embeddings = self.get_memories_with_embeddings()

        if not memories_with_embeddings:
            return []

        # Calculate similarity for each memory
        memory_similarities = []
        for memory, embedding in memories_with_embeddings:
            # Convert embedding bytes to list of floats
            embedding_vector = bytes_to_embedding(embedding.embedding)

            # Calculate cosine similarity
            similarity = calculate_similarity(query_embedding, embedding_vector)

            # Store memory with its similarity score
            memory_similarities.append((memory, similarity))

        # Sort by similarity score (highest first)
        memory_similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            memory
            for memory, _ in memory_similarities[:limit]
            if _ > RELEVANCE_THRESHOLD
        ]

    def update_memory(self, memory: Memory) -> None:
        """
        Update an existing memory in the SQLite database.

        Args:
            memory: The Memory object to update
        """
        # This is essentially the same as store_memory, since we're using merge
        self.store_memory(memory)

    def fetch_embedding(
        self,
        model: str,
        source_type: str,
        source_id: int,
    ) -> Optional[Embedding]:
        """
        Fetch an embedding from the SQLite database.

        Args:
            model: The model name
            source_type: The type of source (e.g., "memories")
            source_id: The ID of the source

        Returns:
            The Embedding object if found, None otherwise
        """
        with Session(self.engine) as session:
            statement = select(Embedding).where(
                Embedding.source_type == source_type,
                Embedding.source_id == source_id,
                Embedding.model_name == model,
            )
            return session.exec(statement).first()

    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if the memory was deleted, False otherwise
        """
        deleted = False

        # Delete from SQLModel table
        with Session(self.engine) as session:
            stored_memory = session.get(Memory, memory_id)
            if stored_memory:
                session.delete(stored_memory)
                session.commit()
                deleted = True

        # Delete the corresponding embedding
        with Session(self.engine) as session:
            statement = select(Embedding).where(
                Embedding.source_type == Memory.__name__,
                Embedding.source_id == memory_id,
            )
            embedding_table = session.exec(statement).first()
            if embedding_table:
                session.delete(embedding_table)
                session.commit()

        # Delete from vector index
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM memories_index WHERE id = ?", (memory_id,))
            conn.commit()
        except sqlite3.OperationalError:
            # For testing purposes, ignore errors
            logger.debug(
                "Vector operations not available, skipping vector index deletion"
            )
        finally:
            conn.close()

        return deleted

    def store_message(self, message: AllMessageValues) -> int:
        """
        Store a message in the SQLite database.

        Args:
            message: The Message object to store
        """

        # Create SQLModel object
        # Insert or update in database
        with Session(self.engine) as session:
            stored_message = session.merge(Message(data_str=json.dumps(message)))
            session.commit()
            session.refresh(stored_message)
            assert stored_message.id is not None, "Failed to store message"
            return stored_message.id

    def store_messages(self, messages: List[AllMessageValues]) -> List[int]:
        """
        Store multiple messages in the SQLite database.

        Args:
            messages: List of messages to store

        Returns:
            List of message IDs for the stored messages
        """
        message_ids = []
        for message in messages:
            message_id = self.store_message(message)
            message_ids.append(message_id)
        return message_ids

    def get_message(self, message_id: int) -> Optional[AllMessageValues]:
        """
        Retrieve a message by ID.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            The Message object if found, None otherwise
        """
        with Session(self.engine) as session:
            stored_message = session.get(Message, message_id)

            if not stored_message:
                return None

            return stored_message.data

    def store_message_set(self, message_set: MessageSet) -> int:
        """
        Store a message set in the SQLite database.

        Args:
            message_set: The MessageSet object to store
        """
        # Convert message_ids list to JSON string

        # Ensure user_id is set
        if not hasattr(message_set, "user_id") or message_set.user_id is None:
            message_set.user_id = self.get_default_user_id()

        # Insert or update in database
        with Session(self.engine) as session:
            message_set = session.merge(message_set)
            session.commit()
            session.refresh(message_set)
            assert message_set.id is not None, "Failed to store message set"
            return message_set.id

    def get_message_set(self, message_set_id: int) -> Optional[MessageSet]:
        """
        Retrieve a message set by ID.

        Args:
            message_set_id: The ID of the message set to retrieve

        Returns:
            The MessageSet object if found, None otherwise
        """
        with Session(self.engine) as session:
            return session.get(MessageSet, message_set_id)

    def get_active_message_set_id(self) -> int:
        message_set = self.get_active_message_set()
        assert message_set and message_set.id
        return message_set.id

    def get_active_message_ids(self) -> List[int]:
        message_set = self.get_active_message_set()
        assert message_set
        return message_set.message_ids

    def get_active_message_set(self) -> Optional[MessageSet]:
        """
        Retrieve the active message set.

        Returns:
            The active MessageSet object if found, None otherwise
        """
        with Session(self.engine) as session:
            statement = (
                select(MessageSet)
                .where(MessageSet.active == True)
                .order_by(desc(MessageSet.created_at))
                .limit(1)
            )
            results = session.exec(statement).all()

            if not results:
                return None

            return results[0]

    def get_messages_in_set(self, message_set_id: int) -> List[AllMessageValues]:
        """
        Retrieve all messages in a message set.

        Args:
            message_set_id: The ID of the message set

        Returns:
            List of Message objects in the message set
        """
        message_set = self.get_message_set(message_set_id)
        if not message_set:
            return []

        with Session(self.engine) as session:
            messages = []
            for message_id in message_set.message_ids:
                stored_message = session.get(Message, message_id)
                assert stored_message
                messages.append(stored_message.data)

            return messages

    def deactivate_all_message_sets(self) -> None:
        """
        Deactivate all message sets.
        """
        with Session(self.engine) as session:
            statement = select(MessageSet)
            stored_message_sets = session.exec(statement).all()

            for stored_message_set in stored_message_sets:
                stored_message_set.active = False

            session.commit()

    def _create_default_user(self) -> None:
        """Create the default user if it doesn't exist."""
        with Session(self.engine) as session:
            # Check if default user exists
            statement = select(User).where(User.token == "default")
            default_user = session.exec(statement).first()

            if not default_user:
                # Create default user with ID 1
                default_user = User(id=1, token="default")
                session.add(default_user)
                session.commit()

    def create_user(self, token: str) -> int:
        """
        Create a new user with the given token.

        Args:
            token: The unique token for the user

        Returns:
            The ID of the created user
        """
        with Session(self.engine) as session:
            # Check if user with this token already exists
            existing_user = session.exec(
                select(User).where(User.token == token)
            ).first()

            if existing_user:
                id = existing_user.id
            else:
                new_user = User(token=token)
                session.add(new_user)
                session.commit()
                session.refresh(new_user)
                id = new_user.id

            assert id
            return id

    def get_user_by_token(self, token: Optional[str]) -> Optional[int]:
        """
        Get a user ID by token.

        Args:
            token: The token to look up

        Returns:
            The user ID if found, None otherwise
        """
        if token is None:
            return None

        with Session(self.engine) as session:
            statement = select(User).where(User.token == token)
            user = session.exec(statement).first()

            if not user:
                return None

            return user.id

    def get_default_user_id(self) -> int:
        """
        Get the ID of the default user.

        Returns:
            The ID of the default user (should be 1)
        """
        user_id = self.get_user_by_token("default")
        if user_id is None:
            # Create default user if it doesn't exist
            self._create_default_user()
            user_id = self.get_user_by_token("default")

        assert user_id

        return user_id

    def close(self) -> None:
        """
        Close any open connections to the database.

        This method is called during test cleanup to ensure proper resource management.
        """
        # Currently, we don't maintain any persistent connections that need to be closed
        # The engine and sessions are managed with context managers
        # This method exists to satisfy the test fixture expectations
