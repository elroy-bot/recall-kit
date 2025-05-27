"""
Command-line interface for Recall Kit.

This module provides a CLI for interacting with Recall Kit, including commands
for creating, searching, and managing memories.
"""

import logging
import sys
from typing import List, Optional

import click
from litellm import AllMessageValues

from recall_kit import __version__
from recall_kit.storage import SQLiteBackend

from .constants import DEFAULT_USER_TOKEN
from .core import RecallKit
from .utils.completion import extract_content_from_response
from .utils.messaging import to_assistant_message, to_system_message, to_user_message


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--storage-type",
    type=click.Choice(["sqlite"]),
    default="sqlite",
    help="Storage backend type",
)
@click.option(
    "--connection-string",
    help="Database connection string",
)
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    help="Embedding model to use",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode",
)
@click.pass_context
def cli(
    ctx: click.Context,
    storage_type: str,
    connection_string: Optional[str],
    embedding_model: str,
    debug: bool,
):
    """Recall Kit: Lightweight memory integrations for LLMs."""
    # Initialize storage backend
    if storage_type == "sqlite":
        storage = SQLiteBackend(connection_string)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

    if debug:
        import litellm

        litellm._turn_on_debug()  # type: ignore
        click.echo("Debug mode enabled")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Initialize RecallKit with storage and default functions
    ctx.ensure_object(dict)
    ctx.obj["recall"] = RecallKit(
        embedding_model=embedding_model, storage=storage, user_token=DEFAULT_USER_TOKEN
    )


@cli.command()
@click.argument("text")
@click.option("--title", help="Title for the memory")
@click.option("--tags", help="Comma-separated list of tags")
@click.pass_context
def remember(
    ctx: click.Context,
    text: str,
    title: Optional[str] = None,
    tags: Optional[str] = None,
):
    """Create a new memory."""
    recall: RecallKit = ctx.obj["recall"]

    # Create the memory
    memory = recall.memory_store.create_memory(
        text=text,
        title=title or text[:50],
    )

    # Print the result
    click.echo(f"Memory created: ID {memory.id}")
    click.echo(f"Title: {memory.title}")
    click.echo(f"Text: {memory.content}")
    if tags:
        click.echo(f"Tags: {tags}")


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Maximum number of results to return")
@click.option("--json", is_flag=True, help="Output results as JSON")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, json: bool):
    """Search for memories."""
    recall: RecallKit = ctx.obj["recall"]

    # Search for memories
    memories = recall.memory_store.search_memories(query, limit=limit)

    if json:
        import json as json_lib

        # Convert memories to dictionaries
        memory_dicts = []
        for memory in memories:
            memory_dict = {
                "id": memory.id,
                "title": memory.title,
                "content": memory.content,
                "created_at": str(memory.created_at),
                "source_metadata": memory.source_metadata,
            }
            memory_dicts.append(memory_dict)
        click.echo(json_lib.dumps(memory_dicts))
    else:
        click.echo(f"Found {len(memories)} memories:")
        for i, memory in enumerate(memories):
            click.echo(f"\n{i+1}. {memory.title}")
            click.echo(f"   {memory.content}")
            # Extract tags from source_metadata if available
            tags = []
            for metadata in memory.source_metadata:
                if isinstance(metadata, dict) and "tags" in metadata:
                    tags.extend(metadata["tags"])
            if tags:
                click.echo(f"   Tags: {', '.join(tags)}")


@cli.command()
@click.option("--threshold", default=0.7, help="Similarity threshold for clustering")
@click.option("--min-cluster", default=2, help="Minimum cluster size")
@click.option("--model", default="gpt-4o", help="Model to use for consolidation")
@click.pass_context
def consolidate(ctx: click.Context, threshold: float, min_cluster: int, model: str):
    """Consolidate similar memories."""
    recall: RecallKit = ctx.obj["recall"]

    # Consolidate memories
    result = recall.memory_consolidator.consolidate_memories(
        completion_model=model,
        threshold=threshold,
        min_cluster_size=min_cluster,
    )

    click.echo(f"Created {len(result)} consolidated memories")


@cli.command()
@click.option("--model", default="gpt-4o", help="Model to use for chat")
@click.pass_context
def chat(ctx: click.Context, model: str):
    """Start an interactive chat session."""

    recall: RecallKit = ctx.obj["recall"]

    click.echo("Recall Kit Chat")
    click.echo(f"Using model: {model}")
    click.echo("Type 'exit' to quit, 'help' for commands")

    # Initialize chat history
    messages: List[AllMessageValues] = [
        to_system_message(
            "You are a helpful assistant with access to the user's memories.",
        )
    ]

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")

            # Check for special commands
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "help":
                click.echo("\nCommands:")
                click.echo("  help - Show this help message")
                click.echo("  exit - Exit the chat")
                click.echo("  clear - Clear the conversation history")
                click.echo("  remember <text> - Create a new memory")
                click.echo("  search <query> - Search for memories")
                continue
            elif user_input.lower() == "clear":
                messages = [messages[0]]  # Keep only the system message
                click.echo("Conversation history cleared")
                continue
            elif user_input.lower().startswith("remember "):
                memory_text = user_input[9:]
                memory = recall.memory_store.create_memory(
                    text=memory_text,
                    title=memory_text[:50],
                )
                click.echo(f"Memory created: ID {memory.id}")
                continue
            elif user_input.lower().startswith("search "):
                query = user_input[7:]
                memories = recall.memory_store.search_memories(query, limit=5)
                click.echo(f"Found {len(memories)} memories:")
                for i, memory in enumerate(memories):
                    click.echo(f"\n{i+1}. {memory.title}")
                    click.echo(f"   {memory.content}")
                continue

            # Add user message to history
            messages.append(to_user_message(user_input))

            # Get response from model
            response = recall.completion(
                model=model,
                messages=messages,
                temperature=0.7,
            )

            # Extract assistant message content safely
            content = extract_content_from_response(response)

            # Add assistant message to history
            if content:
                messages.append(to_assistant_message(content))
                # Print response
                click.echo(f"\nAssistant: {content}")
            else:
                click.echo("\nError: Could not extract assistant message from response")

        except KeyboardInterrupt:
            click.echo("\nExiting...")
            break
        except Exception as e:
            click.echo(f"\nError: {e}")
            continue


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option(
    "--model",
    default="gpt-4o",
    help="LLM model to use for completions and consolidation",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, model: str):
    """Start a memory server."""
    click.echo(f"Starting memory server on http://{host}:{port}...")
    click.echo("Press Ctrl+C to stop the server.")

    # Import here to avoid requiring fastapi and uvicorn for non-server use
    import uvicorn

    from recall_kit.server import create_app

    # Get the RecallKit instance from the context
    recall: RecallKit = ctx.obj["recall"]

    # Create the app with the RecallKit instance
    app = create_app(
        recall=recall,
    )

    # Run the server
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
