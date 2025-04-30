"""
Tests for the core functionality of Recall Kit.
"""

import datetime
import pytest


from recall_kit import Memory, MemorySource


def test_memory_creation():
    """Test creating a Memory object."""
    memory = Memory(
        text="This is a test memory",
        title="Test Memory",
        user_id=1,
    )

    assert memory.text == "This is a test memory"
    assert memory.title == "Test Memory"
    assert memory.embedding is None
    assert memory.source_address is None
    assert memory.parent_ids == []
    assert isinstance(memory.created_at, datetime.datetime)
    assert memory.relevance == 0.0
    assert memory.metadata == {}

def test_memory_validation():
    """Test memory validation."""
    # Valid embedding
    memory = Memory(
        text="Test",
        title="Test",
        embedding=[0.1, 0.2, 0.3],
        user_id=1,
    )
    assert memory.embedding == [0.1, 0.2, 0.3]

    # Invalid embedding (not a list)
    with pytest.raises(ValueError):
        Memory(
            text="Test",
            title="Test",
            embedding="not a list",
            user_id=1,
        )

    # Invalid embedding (not all numbers)
    with pytest.raises(ValueError):
        Memory(
            text="Test",
            title="Test",
            embedding=[0.1, "not a number", 0.3],
            user_id=1,
        )


def test_memory_to_dict():
    """Test converting a Memory to a dictionary."""
    memory = Memory(
        text="This is a test memory",
        title="Test Memory",
        embedding=[0.1, 0.2, 0.3],
        source_address="test:123",
        metadata={"key": "value"},
        user_id=1,
    )

    memory_dict = memory.to_dict()

    assert memory_dict["text"] == "This is a test memory"
    assert memory_dict["title"] == "Test Memory"
    assert memory_dict["source_address"] == "test:123"
    assert memory_dict["metadata"] == {"key": "value"}
    assert "embedding" not in memory_dict


def test_memory_source_creation():
    """Test creating a MemorySource object."""
    source = MemorySource(
        text="This is a test source",
        title="Test Source",
        address="test:123",
        user_id=1,
        metadata={"key": "value"},
    )

    assert source.text == "This is a test source"
    assert source.title == "Test Source"
    assert source.address == "test:123"
    assert source.metadata == {"key": "value"}


def test_memory_source_to_memory():
    """Test converting a MemorySource to a Memory."""
    source = MemorySource(
        text="This is a test source",
        title="Test Source",
        address="test:123",
        metadata={"key": "value"},
        user_id=1,
    )

    memory = source.to_memory()

    assert memory.text == "This is a test source"
    assert memory.title == "Test Source"
    assert memory.source_address == "test:123"
    assert memory.metadata == {"key": "value"}
    assert memory.embedding is None




def test_create_memory(recall_kit):
    """Test creating a memory."""
    memory = recall_kit.create_memory(
        text="This is a test memory",
        title="Test Memory",
        source_address="test:123",
        metadata={"key": "value"},
    )

    assert memory.text == "This is a test memory"
    assert memory.title == "Test Memory"
    assert memory.source_address == "test:123"
    assert memory.metadata == {"key": "value"}
    assert memory.embedding is not None

    # Verify the memory was stored
    retrieved = recall_kit.storage.get_memory(memory.id)
    assert retrieved.text == "This is a test memory"

def test_add_memory(recall_kit):
    """Test adding an existing memory."""
    memory = Memory(
        text="This is a test memory",
        title="Test Memory",
        source_address="test:123",
        metadata={"key": "value"},
        user_id=1,
    )

    added = recall_kit.add_memory(memory)

    assert added.id == memory.id
    assert added.text == "This is a test memory"
    assert added.embedding is not None

    # Verify the memory was stored
    retrieved = recall_kit.storage.get_memory(memory.id)
    assert retrieved.text == "This is a test memory"


def test_search(recall_kit):
    """Test searching for memories."""
    # Create some test memories
    recall_kit.create_memory(
        text="The cat sat on the mat",
        title="Cat Memory",
    )
    recall_kit.create_memory(
        text="The dog chased the ball",
        title="Dog Memory",
    )
    recall_kit.create_memory(
        text="The bird flew in the sky",
        title="Bird Memory",
    )

    # Search for cat-related memories
    results = recall_kit.search("cat", limit=2)

    assert len(results) == 2
    assert any("cat" in memory.text.lower() for memory in results)

    # Check that relevance scores are set
    for memory in results:
        assert memory.relevance >= 0.0

def test_consolidate_memories(recall_kit):
    """Test consolidating memories."""
    # Create similar memories
    recall_kit.create_memory(
        text="The cat sat on the mat",
        title="Cat Memory 1",
    )
    recall_kit.create_memory(
        text="The cat was sitting on the mat",
        title="Cat Memory 2",
    )
    recall_kit.create_memory(
        text="A feline was on a floor covering",
        title="Cat Memory 3",
    )

    # Create a different memory
    recall_kit.create_memory(
        text="The dog chased the ball",
        title="Dog Memory",
    )

    # Consolidate memories
    consolidated = recall_kit.consolidate_memories(
        threshold=0.5,  # Lower threshold for testing
        min_cluster_size=2,
        max_cluster_size=3,
    )

    # Should have created at least one consolidated memory
    assert len(consolidated) >= 1

    # Check the consolidated memory
    for memory in consolidated:
        assert memory.parent_ids
        assert "Consolidated" in memory.title
        assert "Consolidated memory:" in memory.text
        assert memory.metadata.get("consolidated") is True

def test_compress_messages(recall_kit):
    """Test compressing messages."""
    import datetime

    # Create a list of messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Tell me about the weather."},
        {"role": "assistant", "content": "I don't have real-time weather information."},
        {"role": "user", "content": "What about climate change?"},
        {"role": "assistant", "content": "Climate change is a significant global issue..."},
        {"role": "user", "content": "And renewable energy?"},
        {"role": "assistant", "content": "Renewable energy sources include solar, wind..."},
    ]

    # Set a low token count to force compression
    compressed = recall_kit.compress_messages(messages, target_token_count=100)

    # Check that the system message is preserved
    assert any(msg.get("role") == "system" for msg in compressed)

    # Check that we have fewer messages than we started with
    assert len(compressed) < len(messages)

    # Check that the most recent messages are preserved
    assert compressed[-1].get("content") == "Renewable energy sources include solar, wind..."
    assert compressed[-2].get("content") == "And renewable energy?"

    # Check that a memory was created from the dropped messages
    memories = recall_kit.storage.get_all_memories()
    compressed_memories = [m for m in memories if m.metadata.get("compressed") is True]
    assert len(compressed_memories) > 0

def test_compress_messages_with_age_limit(recall_kit):
    """Test compressing messages with an age limit."""
    import datetime

    # Create messages with timestamps
    now = datetime.datetime.now()
    one_day_ago = now - datetime.timedelta(days=1)
    two_days_ago = now - datetime.timedelta(days=2)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Old message", "created_at": two_days_ago.isoformat()},
        {"role": "assistant", "content": "Old response", "created_at": two_days_ago.isoformat()},
        {"role": "user", "content": "Recent message", "created_at": one_day_ago.isoformat()},
        {"role": "assistant", "content": "Recent response", "created_at": one_day_ago.isoformat()},
        {"role": "user", "content": "Latest message", "created_at": now.isoformat()},
        {"role": "assistant", "content": "Latest response", "created_at": now.isoformat()},
    ]

    # Compress with a 1.5 day age limit
    compressed = recall_kit.compress_messages(
        messages,
        target_token_count=10000,  # High token count so age is the limiting factor
        max_message_age=datetime.timedelta(days=1.5)
    )

    # Check that old messages are dropped
    assert len(compressed) < len(messages)
    assert not any(msg.get("content") == "Old message" for msg in compressed)
    assert not any(msg.get("content") == "Old response" for msg in compressed)

    # Check that recent and latest messages are kept
    assert any(msg.get("content") == "Recent message" for msg in compressed)
    assert any(msg.get("content") == "Recent response" for msg in compressed)
    assert any(msg.get("content") == "Latest message" for msg in compressed)
    assert any(msg.get("content") == "Latest response" for msg in compressed)

def test_compress_messages_tool_calls(recall_kit):
    """Test compressing messages with tool calls."""

    # Create messages with tool calls
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "I'll check the weather for you."},
        {"role": "tool", "content": "The weather is sunny and 75°F."},
        {"role": "user", "content": "Thanks! What about tomorrow?"},
        {"role": "assistant", "content": "I'll check tomorrow's forecast."},
        {"role": "tool", "content": "Tomorrow will be partly cloudy with a high of 70°F."},
    ]

    # Compress with a low token count
    compressed = recall_kit.compress_messages(messages, target_token_count=50)

    # Check that tool calls and their corresponding assistant messages are kept together
    for i, msg in enumerate(compressed):
        if msg.get("role") == "tool":
            # The previous message should be an assistant message
            assert i > 0
            assert compressed[i-1].get("role") == "assistant"
