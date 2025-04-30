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


def test_search_command(cli_runner: CliRunner, temp_db):
    """Test the 'search' command."""
    # First create a memory
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "The cat sat on the mat",
            "--title",
            "Cat Memory",
        ],
    )

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


def test_search_json_output(cli_runner: CliRunner, temp_db):
    """Test the 'search' command with JSON output."""
    # First create a memory
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "The dog chased the ball",
            "--title",
            "Dog Memory",
        ],
    )

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
    assert output[0]["text"] == "The dog chased the ball"


def test_consolidate_command(cli_runner: CliRunner, temp_db):
    """Test the 'consolidate' command."""
    # Create similar memories
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "The cat sat on the mat",
            "--title",
            "Cat Memory 1",
        ],
    )
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "The cat was sitting on the mat",
            "--title",
            "Cat Memory 2",
        ],
    )

    # Consolidate memories
    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "consolidate",
            "--threshold",
            "0.5",
            "--min-cluster",
            "2",
            "--model",
            "gpt-4o",  # Add model parameter
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Created" in result.output


def test_export_command(cli_runner: CliRunner, temp_db):
    """Test the 'export' command."""
    # Create a memory
    cli_runner.invoke(
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

    # Create a temporary file for export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_file = os.path.join(temp_dir, "export.json")
        result = cli_runner.invoke(
            cli,
            [
                "--connection-string",
                temp_db,
                "export",
                "--format",
                "json",
                "--output",
                export_file,
            ],
        )

        assert result.exit_code == 0
        assert "Exported" in result.output

        # Check the exported file
        with open(export_file, "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["title"] == "Test Memory"
            assert data[0]["text"] == "This is a test memory"


def test_stats_command(cli_runner: CliRunner, temp_db):
    """Test the 'stats' command."""
    # Create some memories
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "Memory 1",
            "--title",
            "Memory 1",
        ],
    )
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "Memory 2",
            "--title",
            "Memory 2",
        ],
    )

    # Get stats
    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "stats",
        ],
    )

    assert result.exit_code == 0
    assert "Total memories: 2" in result.output


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
def test_chat_search_command(mock_input, cli_runner: CliRunner, temp_db):
    """Test the 'search' command in chat mode."""
    # First create a memory
    cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "remember",
            "The bird flew in the sky",
            "--title",
            "Bird Memory",
        ],
    )

    # Mock user input to use search command and then exit
    mock_input.side_effect = ["search bird", "exit"]

    result = cli_runner.invoke(
        cli,
        [
            "--connection-string",
            temp_db,
            "chat",
        ],
    )

    assert result.exit_code == 0
    assert "Found" in result.output
    assert "Bird Memory" in result.output


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
                            "obj", (object,), {"content": "This is a test response"}
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
    assert kwargs["messages"][0]["role"] == "system"
    # Second message should be user
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["messages"][1]["content"] == "What is the meaning of life?"
    # Third message should be assistant (from a previous interaction or initialization)
    assert kwargs["messages"][2]["role"] == "assistant"
