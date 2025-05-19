"""
Tests for the CLI functionality of Recall Kit.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from recall_kit.cli import cli
from recall_kit.constants import CONTENT, ROLE, SYSTEM, USER


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test.db")
    yield db_path
    temp_dir.cleanup()


def test_remember_command(cli_runner: CliRunner, temp_db):
    """Test the 'remember' command."""
    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "This is a test memory",
            "--title",
            "Test Memory",
        ],
    )

    assert result.exit_code == 0
    assert "Memory created:" in result.output
    assert "Title: Test Memory" in result.output


def test_search_command(cli_runner: CliRunner, temp_db, monkeypatch):
    """Test the 'search' command."""
    # Mock the search_memories method to return a test memory
    import datetime

    from recall_kit.models import Memory
    from recall_kit.services.memory import MemoryService

    def mock_search_memories(self, query, limit=5):
        # Return a test memory
        return [
            Memory(
                id=1,
                content="The cat sat on the mat",
                title="Cat Memory",
                user_id=1,
                created_at=datetime.datetime.now(),
                _source_metadata="[]",
            )
        ]

    # Apply the monkey patch
    monkeypatch.setattr(MemoryService, "search_memories", mock_search_memories)

    # Then search for it
    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "search",
            "cat",
        ],
    )

    assert result.exit_code == 0
    assert "Found" in result.output
    assert "Cat Memory" in result.output
    assert "The cat sat on the mat" in result.output


def test_search_json_output(cli_runner: CliRunner, temp_db, monkeypatch):
    """Test the 'search' command with JSON output."""
    # Mock the search_memories method to return a test memory
    import datetime

    from recall_kit.models import Memory
    from recall_kit.services.memory import MemoryService

    def mock_search_memories(self, query, limit=5):
        # Return a test memory
        return [
            Memory(
                id=1,
                content="The dog chased the ball",
                title="Dog Memory",
                user_id=1,
                created_at=datetime.datetime.now(),
                _source_metadata="[]",
            )
        ]

    # Apply the monkey patch
    monkeypatch.setattr(MemoryService, "search_memories", mock_search_memories)

    # Then search for it with JSON output
    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "search",
            "dog",
            "--json",
        ],
    )

    assert result.exit_code == 0

    # Parse the JSON output
    output = json.loads(result.output)
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0]["title"] == "Dog Memory"
    assert output[0]["content"] == "The dog chased the ball"


@patch("recall_kit.cli.input")
def test_chat_command(mock_input, cli_runner: CliRunner, temp_db):
    """Test the 'chat' command."""
    # Mock user input to exit after one interaction
    mock_input.return_value = "exit"

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
            "--model",
            "gpt-3.5-turbo",
        ],
    )

    assert (
        result.exit_code == 0
    ), f"Exit code: {result.exit_code}, Output: {result.output.replace('\n', ' ')}"
    assert "Recall Kit Chat" in result.output
    assert "Using model: gpt-3.5-turbo" in result.output


@patch("recall_kit.cli.input")
def test_chat_help_command(mock_input, cli_runner: CliRunner, temp_db):
    """Test the 'help' command in chat mode."""
    # Mock user input to use help command and then exit
    mock_input.side_effect = ["help", "exit"]

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
        ],
    )

    assert result.exit_code == 0
    assert "Commands:" in result.output
    assert "clear - Clear the conversation history" in result.output


@patch("recall_kit.cli.input")
def test_chat_remember_command(mock_input, cli_runner: CliRunner, temp_db):
    """Test the 'remember' command in chat mode."""
    # Mock user input to use remember command and then exit
    mock_input.side_effect = ["remember This is a memory from chat", "exit"]

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
        ],
    )

    assert result.exit_code == 0
    assert "Memory created:" in result.output


@patch("recall_kit.cli.input")
def test_chat_clear_command(mock_input, cli_runner: CliRunner, temp_db):
    """Test the 'clear' command in chat mode."""
    # Mock user input to use clear command and then exit
    mock_input.side_effect = ["clear", "exit"]

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
        ],
    )

    assert result.exit_code == 0
    assert "Conversation history cleared" in result.output


@patch("recall_kit.cli.input")
@patch("recall_kit.core.RecallKit.completion")
def test_chat_completion(
    mock_chat_completion, mock_input, cli_runner: CliRunner, temp_db
):
    """Test chat completion functionality."""
    # Mock the chat_completion method to return a response
    mock_response = type(
        "obj",
        (object,),
        {
            "choices": [
                type(
                    "obj",
                    (object,),
                    {
                        "message": type(
                            "obj", (object,), {CONTENT: "This is a test response"}
                        )
                    },
                )
            ]
        },
    )
    mock_chat_completion.return_value = mock_response

    # Mock user input to ask a question and then exit
    mock_input.side_effect = ["What is the meaning of life?", "exit"]

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
            "--model",
            "gpt-3.5-turbo",
        ],
    )

    assert result.exit_code == 0
    # The output is printed to stdout, not captured in result.output
    # Just check that the chat command ran successfully
    assert "Recall Kit Chat" in result.output
    assert "Using model: gpt-3.5-turbo" in result.output

    # Verify that chat_completion was called with the correct arguments
    mock_chat_completion.assert_called_once()
    args, kwargs = mock_chat_completion.call_args
    assert kwargs["model"] == "gpt-3.5-turbo"
    # Check that we have the expected messages
    assert (
        len(kwargs["messages"]) == 3
    )  # System message + user message + assistant message
    # First message should be system
    assert kwargs["messages"][0][ROLE] == SYSTEM
    # Second message should be user
    assert kwargs["messages"][1][ROLE] == USER
    assert kwargs["messages"][1][CONTENT] == "What is the meaning of life?"
    # Third message should be assistant (from a previous interaction or initialization)
    assert kwargs["messages"][2][ROLE] == "assistant"
