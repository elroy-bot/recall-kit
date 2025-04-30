"""
Tests for the smolagents integration of Recall Kit.
"""

from unittest.mock import MagicMock, patch

try:
    import smolagents
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

import pytest

from recall_kit import RecallKit, Memory


@pytest.fixture
def mock_recall_kit():
    """Create a mock RecallKit instance for testing."""
    recall_kit = MagicMock(spec=RecallKit)

    # Mock the search method to return some test memories
    memory1 = Memory(
        text="This is memory 1",
        title="Memory 1",
    )
    memory1.relevance = 0.9

    memory2 = Memory(
        text="This is memory 2",
        title="Memory 2",
    )
    memory2.relevance = 0.8

    recall_kit.search.return_value = [memory1, memory2]

    # Mock the create_memory method to return a test memory
    created_memory = Memory(
        text="This is a created memory",
        title="Created Memory",
    )
    recall_kit.create_memory.return_value = created_memory

    # Mock the consolidate_memories method to return an empty list
    recall_kit.consolidate_memories.return_value = []

    return recall_kit


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
@patch("recall_kit.smolagents.RecallMemoryTool")
def test_recall_memory_tool_import(mock_tool):
    """Test that RecallMemoryTool can be imported."""
    from recall_kit.smolagents import RecallMemoryTool
    assert RecallMemoryTool == mock_tool


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
@patch("recall_kit.smolagents.CreateMemoryTool")
def test_create_memory_tool_import(mock_tool):
    """Test that CreateMemoryTool can be imported."""
    from recall_kit.smolagents import CreateMemoryTool
    assert CreateMemoryTool == mock_tool


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
@patch("recall_kit.smolagents.RecallKitAgentMemory")
def test_recall_kit_agent_memory_import(mock_memory):
    """Test that RecallKitAgentMemory can be imported."""
    from recall_kit.smolagents import RecallKitAgentMemory
    assert RecallKitAgentMemory == mock_memory

@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_recall_memory_tool(mock_recall_kit):
    """Test the RecallMemoryTool."""
    from recall_kit.smolagents import RecallMemoryTool

    # Create the tool
    tool = RecallMemoryTool(recall_kit=mock_recall_kit)

    # Run the tool
    result = tool.run("test query")

    # Check that the recall_kit.search method was called
    mock_recall_kit.search.assert_called_once_with("test query", limit=3)

    # Check the result
    assert result["query"] == "test query"
    assert result["count"] == 2
    assert len(result["memories"]) == 2
    assert result["memories"][0]["title"] == "Memory 1"
    assert result["memories"][0]["text"] == "This is memory 1"
    assert result["memories"][0]["relevance"] == 0.9
    assert result["memories"][1]["title"] == "Memory 2"
    assert result["memories"][1]["text"] == "This is memory 2"
    assert result["memories"][1]["relevance"] == 0.8


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_recall_memory_tool_with_relevance_threshold(mock_recall_kit):
    """Test the RecallMemoryTool with a relevance threshold."""
    from recall_kit.smolagents import RecallMemoryTool

    # Create the tool with a high relevance threshold
    tool = RecallMemoryTool(
        recall_kit=mock_recall_kit,
        relevance_threshold=0.85,
    )

    # Run the tool
    result = tool.run("test query")

    # Check the result (only the first memory should be included)
    assert result["count"] == 1
    assert len(result["memories"]) == 1
    assert result["memories"][0]["title"] == "Memory 1"

@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_create_memory_tool(mock_recall_kit):
    """Test the CreateMemoryTool."""
    from recall_kit.smolagents import CreateMemoryTool

    # Create the tool
    tool = CreateMemoryTool(recall_kit=mock_recall_kit)

    # Run the tool
    result = tool.run(
        text="This is a test memory",
        title="Test Memory",
        importance=0.7,
    )

    # Check that the recall_kit.create_memory method was called
    mock_recall_kit.create_memory.assert_called_once_with(
        text="This is a test memory",
        title="Test Memory",
        metadata={"importance": 0.7},
    )

    # Check the result
    assert result["created"] is True
    assert result["memory_id"] == mock_recall_kit.create_memory.return_value.id
    assert result["title"] == "Created Memory"
    assert result["consolidated_count"] == 0


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_create_memory_tool_with_auto_consolidate(mock_recall_kit):
    """Test the CreateMemoryTool with auto_consolidate=True."""
    from recall_kit.smolagents import CreateMemoryTool

    # Create the tool with auto_consolidate=True
    tool = CreateMemoryTool(
        recall_kit=mock_recall_kit,
        auto_consolidate=True,
    )

    # Run the tool
    result = tool.run(
        text="This is a test memory",
        title="Test Memory",
    )

    # Check that the recall_kit.consolidate_memories method was called
    mock_recall_kit.consolidate_memories.assert_called_once()

@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_create_memory_tool_with_importance_threshold(mock_recall_kit):
    """Test the CreateMemoryTool with an importance threshold."""
    from recall_kit.smolagents import CreateMemoryTool

    # Create the tool with an importance threshold
    tool = CreateMemoryTool(
        recall_kit=mock_recall_kit,
        importance_threshold=0.8,
    )

    # Run the tool with an importance below the threshold
    result = tool.run(
        text="This is a test memory",
        title="Test Memory",
        importance=0.7,
    )

    # Check that the memory was not created
    mock_recall_kit.create_memory.assert_not_called()
    assert result["created"] is False
    assert "below threshold" in result["reason"]


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_recall_kit_agent_memory(mock_recall_kit):
    """Test the RecallKitAgentMemory."""
    from recall_kit.smolagents import RecallKitAgentMemory

    # Create the agent memory
    agent_memory = RecallKitAgentMemory(recall_kit=mock_recall_kit)

    # Test adding a memory
    memory_id = agent_memory.add_memory(
        text="This is a test memory",
        title="Test Memory",
    )

    # Check that the recall_kit.create_memory method was called
    mock_recall_kit.create_memory.assert_called_once_with(
        text="This is a test memory",
        title="Test Memory",
        metadata={},
    )

    # Check the memory ID
    assert memory_id == mock_recall_kit.create_memory.return_value.id

    # Test getting relevant memories
    memories = agent_memory.get_relevant_memories("test query")

    # Check that the recall_kit.search method was called
    mock_recall_kit.search.assert_called_with("test query", limit=5)

    # Check the memories
    assert len(memories) == 2
    assert memories[0].title == "Memory 1"
    assert memories[1].title == "Memory 2"

    # Test getting context
    context = agent_memory.get_context("test query")

    # Check the context
    assert "Relevant memories" in context
    assert "1. This is memory 1" in context
    assert "2. This is memory 2" in context

@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_recall_kit_agent_memory_with_auto_consolidate(mock_recall_kit):
    """Test the RecallKitAgentMemory with auto_consolidate=True."""
    from recall_kit.smolagents import RecallKitAgentMemory

    # Create the agent memory with auto_consolidate=True
    agent_memory = RecallKitAgentMemory(
        recall_kit=mock_recall_kit,
        auto_consolidate=True,
        consolidation_interval=2,
    )

    # Add a memory (should not trigger consolidation)
    agent_memory.add_memory(text="Memory 1")
    mock_recall_kit.consolidate_memories.assert_not_called()

    # Add another memory (should trigger consolidation)
    agent_memory.add_memory(text="Memory 2")
    mock_recall_kit.consolidate_memories.assert_called_once()


@pytest.mark.skipif(not SMOLAGENTS_AVAILABLE, reason="smolagents not installed")
def test_recall_kit_agent_memory_process_response(mock_recall_kit):
    """Test the RecallKitAgentMemory.process_response method."""
    from recall_kit.smolagents import RecallKitAgentMemory

    # Create the agent memory
    agent_memory = RecallKitAgentMemory(recall_kit=mock_recall_kit)

    # Process a response
    agent_memory.process_response(
        query="What is the meaning of life?",
        response="42",
    )

    # Check that the recall_kit.create_memory method was called
    mock_recall_kit.create_memory.assert_called_once()
    args, kwargs = mock_recall_kit.create_memory.call_args
    assert "User: What is the meaning of life?" in kwargs["text"]
    assert "Assistant: 42" in kwargs["text"]
    assert kwargs["metadata"] == {"type": "conversation"}
