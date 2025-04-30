"""
Command-line interface for Recall Kit.

This module provides a CLI for interacting with Recall Kit, including commands
for creating, searching, and managing memories.
"""

import sys
import json
from typing import List, Optional, TextIO

import click

from recall_kit import RecallKit, Memory, __version__
from recall_kit.storage import SQLiteBackend


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--storage-type",
    type=click.Choice(["sqlite", "postgres"]),
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
def cli(ctx: click.Context, storage_type: str, connection_string: Optional[str], embedding_model: str, debug: bool):
    """Recall Kit: Lightweight memory integrations for LLMs."""
    # Initialize storage backend
    if storage_type == "sqlite":
        storage = SQLiteBackend(connection_string)
    elif storage_type == "postgres":
        # Import here to avoid requiring postgres dependencies
        from recall_kit.storage import PostgresBackend
        storage = PostgresBackend(connection_string)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

    if debug:
        import litellm
        litellm._turn_on_debug()
        click.echo("Debug mode enabled")

    # Initialize RecallKit with storage and default functions
    ctx.ensure_object(dict)
    ctx.obj["recall"] = RecallKit.create(storage)


@cli.command()
@click.argument("text")
@click.option("--title", help="Title for the memory")
@click.option("--source", help="Source address for the memory")
@click.pass_context
def remember(ctx: click.Context, text: str, title: Optional[str], source: Optional[str]):
    """Create a new memory."""
    recall: RecallKit = ctx.obj["recall"]
    memory: Memory = recall.create_memory(
        text=text,
        title=title,
        source_address=source,
    )
    click.echo(f"Memory created: {memory.id}")
    click.echo(f"Title: {memory.title}")


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Maximum number of results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, output_json: bool):
    """Search for memories."""
    recall: RecallKit = ctx.obj["recall"]
    results: List[Memory] = recall.search(query, limit=limit)

    if output_json:
        # Output as JSON
        output = [
            {
                "id": memory.id,
                "text": memory.text,
                "title": memory.title,
                "relevance": memory.relevance,
                "created_at": memory.created_at.isoformat(),
                "source_address": memory.source_address,
            }
            for memory in results
        ]
        click.echo(json.dumps(output, indent=2))
    else:
        # Output as text
        if not results:
            click.echo("No memories found.")
            return

        click.echo(f"Found {len(results)} memories:")
        for i, memory in enumerate(results, 1):
            click.echo(f"\n{i}. {memory.title} (relevance: {memory.relevance:.2f})")
            click.echo(f"   {memory.text}")
            if memory.source_address:
                click.echo(f"   Source: {memory.source_address}")


@cli.command()
@click.option("--threshold", default=0.85, help="Similarity threshold (0-1)")
@click.option("--min-cluster", default=2, help="Minimum cluster size")
@click.option("--max-cluster", default=5, help="Maximum cluster size")
@click.option("--model", help="LLM model to use for consolidation")
@click.pass_context
def consolidate(ctx: click.Context, model: str, threshold: float, min_cluster: int, max_cluster: int):
    """Consolidate similar memories."""
    recall: RecallKit = ctx.obj["recall"]
    consolidated: List[Memory] = recall.consolidate_memories(
        model=model,
        threshold=threshold,
        min_cluster_size=min_cluster,
        max_cluster_size=max_cluster,
    )

    if not consolidated:
        click.echo("No memories were consolidated.")
        return

    click.echo(f"Created {len(consolidated)} consolidated memories:")
    for i, memory in enumerate(consolidated, 1):
        click.echo(f"\n{i}. {memory.title}")
        click.echo(f"   Parent IDs: {', '.join(memory.parent_ids)}")
        click.echo(f"   {memory.text}")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--include", help="Glob pattern to include files")
@click.option("--exclude", help="Glob pattern to exclude files")
@click.option("--recursive/--no-recursive", default=True, help="Recursively process directories")
@click.pass_context
def ingest(ctx: click.Context, path: str, include: Optional[str], exclude: Optional[str], recursive: bool):
    """Ingest documents from a directory."""
    import glob
    import os

    recall: RecallKit = ctx.obj["recall"]

    # Determine files to process
    if os.path.isfile(path):
        files = [path]
    else:
        # It's a directory
        pattern = os.path.join(path, "**" if recursive else "*")
        if include:
            pattern = os.path.join(pattern, include)
        files = glob.glob(pattern, recursive=recursive)

        # Apply exclude pattern if provided
        if exclude:
            exclude_pattern = os.path.join(path, "**" if recursive else "*", exclude)
            exclude_files = set(glob.glob(exclude_pattern, recursive=recursive))
            files = [f for f in files if f not in exclude_files]

    # Process files
    count = 0
    for file_path in files:
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create a memory for the file
            title = os.path.basename(file_path)
            memory = recall.create_memory(
                text=content,
                title=title,
                source_address=f"file:{file_path}",
                metadata={"file_path": file_path},
            )

            click.echo(f"Ingested: {file_path} -> Memory ID: {memory.id}")
            count += 1

        except Exception as e:
            click.echo(f"Error ingesting {file_path}: {str(e)}", err=True)

    click.echo(f"Ingested {count} files.")


@cli.command()
@click.option("--format", "output_format", type=click.Choice(["json", "csv"]), default="json", help="Output format")
@click.option("--output", type=click.Path(), help="Output file (default: stdout)")
@click.pass_context
def export(ctx: click.Context, output_format: str, output: Optional[str]):
    """Export memories."""
    recall: RecallKit = ctx.obj["recall"]
    memories: List[Memory] = recall.storage.get_all_memories()

    # Prepare output file or use stdout
    output_file: TextIO = open(output, "w", encoding="utf-8") if output else sys.stdout

    try:
        if output_format == "json":
            # Export as JSON
            data = [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "title": memory.title,
                    "created_at": memory.created_at.isoformat(),
                    "source_address": memory.source_address,
                    "parent_ids": memory.parent_ids,
                    "metadata": memory.metadata,
                }
                for memory in memories
            ]
            json.dump(data, output_file, indent=2)

        elif output_format == "csv":
            # Export as CSV
            import csv

            writer = csv.writer(output_file)
            writer.writerow(["id", "title", "text", "created_at", "source_address"])

            for memory in memories:
                writer.writerow([
                    memory.id,
                    memory.title,
                    memory.text,
                    memory.created_at.isoformat(),
                    memory.source_address or "",
                ])

    finally:
        if output:
            output_file.close()

    if output:
        click.echo(f"Exported {len(memories)} memories to {output}")
    else:
        # If we wrote to stdout, add a newline
        if memories:
            click.echo("")


@cli.command()
@click.pass_context
def stats(ctx: click.Context):
    """View memory statistics."""
    recall: RecallKit = ctx.obj["recall"]
    memories: List[Memory] = recall.storage.get_all_memories()

    # Count total memories
    total_count = len(memories)

    # Count consolidated memories
    consolidated_count = sum(1 for m in memories if m.parent_ids)

    # Count source memories
    source_count = sum(1 for m in memories if m.source_address)

    # Calculate average text length
    avg_length = sum(len(m.text) for m in memories) / total_count if total_count else 0

    # Output statistics
    click.echo("Memory Statistics:")
    click.echo(f"Total memories: {total_count}")
    click.echo(f"Consolidated memories: {consolidated_count}")
    click.echo(f"Source-linked memories: {source_count}")
    click.echo(f"Average text length: {avg_length:.1f} characters")


@cli.command()
@click.option("--model", help="LLM model to use for chat")
@click.pass_context
def chat(ctx: click.Context, model: str):
    """Interactive chat with memory and LLM."""
    from litellm import ModelResponse

    recall: RecallKit = ctx.obj["recall"]

    click.echo("Recall Kit Chat")
    click.echo(f"Using model: {model}")
    click.echo("Type 'exit' or 'quit' to end the session")
    click.echo("Type 'help' for commands")

    # Get or create active message set
    active_message_set = recall.get_active_message_set()

    # Initialize messages
    if active_message_set:
        # Load messages from active message set
        messages = recall.get_messages_in_set(active_message_set.id)
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
        click.echo(f"Loaded {len(messages)} messages from active conversation")
    else:
        # Create a new system message
        system_message = recall.create_message(
            role="system",
            content="You are a helpful assistant with access to the user's memories. "
                   "Respond to the user's questions using relevant memories when available.",
            metadata={"type": "conversation"}
        )

        # Create a new message set with the system message
        active_message_set = recall.create_message_set(
            message_ids=[system_message.id],
            active=True,
            metadata={"type": "conversation"}
        )

        messages_dict = [{"role": "system", "content": system_message.content}]
        click.echo("Created new conversation")

    # Start the chat loop
    while True:
        try:
            user_input: str = input("\nYou: ").strip()

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.lower() == "help":
                click.echo("\nCommands:")
                click.echo("  exit, quit - End the session")
                click.echo("  help - Show this help message")
                click.echo("  remember <text> - Create a new memory")
                click.echo("  search <query> - Search for memories")
                click.echo("  clear - Clear the conversation history")
                click.echo("  history - Show conversation history")
                click.echo("  messages - Show all messages in the current conversation")
                continue

            if user_input.lower().startswith("remember "):
                # Create a memory
                text: str = user_input[9:].strip()
                memory: Memory = recall.create_memory(text=text)
                click.echo(f"Memory created: {memory.id}")
                continue

            if user_input.lower().startswith("search "):
                # Search for memories
                query: str = user_input[7:].strip()
                results: List[Memory] = recall.search(query, limit=3)

                if not results:
                    click.echo("No memories found.")
                else:
                    click.echo(f"Found {len(results)} memories:")
                    for i, memory in enumerate(results, 1):
                        click.echo(f"{i}. {memory.title} (relevance: {memory.relevance:.2f})")
                        click.echo(f"   {memory.text}")
                continue

            if user_input.lower() == "clear":
                # Clear conversation history but keep the system message
                system_messages = [msg for msg in recall.get_messages_in_set(active_message_set.id)
                                  if msg.role == "system"]

                if system_messages:
                    # Create a new message set with just the system message
                    active_message_set = recall.create_message_set(
                        message_ids=[msg.id for msg in system_messages],
                        active=True,
                        metadata={"type": "conversation"}
                    )
                    messages_dict = [{"role": msg.role, "content": msg.content} for msg in system_messages]
                else:
                    # Create a new system message
                    system_message = recall.create_message(
                        role="system",
                        content="You are a helpful assistant with access to the user's memories. "
                               "Respond to the user's questions using relevant memories when available.",
                        metadata={"type": "conversation"}
                    )

                    # Create a new message set with the system message
                    active_message_set = recall.create_message_set(
                        message_ids=[system_message.id],
                        active=True,
                        metadata={"type": "conversation"}
                    )

                    messages_dict = [{"role": "system", "content": system_message.content}]

                click.echo("Conversation history cleared.")
                continue

            if user_input.lower() == "history":
                # Show conversation history
                messages = recall.get_messages_in_set(active_message_set.id)
                if not messages:
                    click.echo("No conversation history.")
                else:
                    click.echo("\nConversation history:")
                    for i, msg in enumerate(messages):
                        click.echo(f"{i+1}. {msg.role}: {msg.content}")
                continue

            if user_input.lower() == "messages":
                # Show all messages in the current conversation
                messages = recall.get_messages_in_set(active_message_set.id)
                if not messages:
                    click.echo("No messages in the current conversation.")
                else:
                    click.echo("\nMessages in the current conversation:")
                    for i, msg in enumerate(messages):
                        click.echo(f"{i+1}. ID: {msg.id}")
                        click.echo(f"   Role: {msg.role}")
                        click.echo(f"   Content: {msg.content}")
                        click.echo(f"   Created: {msg.created_at}")
                continue

            # Create a new user message
            user_message = recall.create_message(
                role="user",
                content=user_input,
                metadata={"type": "conversation"}
            )

            # Add the user message to the active message set
            message_ids = active_message_set.message_ids + [user_message.id]
            active_message_set = recall.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"type": "conversation"}
            )

            # Update the messages dictionary for the completion API
            messages_dict.append({"role": "user", "content": user_input})

            # Generate response using chat_completion
            response: ModelResponse = recall.completion(
                messages=messages_dict,
                model=model,
            )

            # Extract assistant's response
            assistant_content: str = response.choices[0].message.content

            # Create a new assistant message
            assistant_message = recall.create_message(
                role="assistant",
                content=assistant_content,
                metadata={"type": "conversation"}
            )

            # Add the assistant message to the active message set
            message_ids = active_message_set.message_ids + [assistant_message.id]
            active_message_set = recall.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"type": "conversation"}
            )

            # Update the messages dictionary for the next iteration
            messages_dict.append({"role": "assistant", "content": assistant_content})

            # Display the response
            click.echo(f"\nAssistant: {assistant_content}")



        except KeyboardInterrupt:
            click.echo("\nExiting...")
            break


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--model", default="gpt-4o", help="LLM model to use for completions and consolidation")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, model: str):
    """Start a memory server."""
    click.echo(f"Starting memory server on http://{host}:{port}...")
    click.echo("Press Ctrl+C to stop the server.")

    # Import here to avoid requiring fastapi and uvicorn for non-server use
    try:
        import uvicorn
        from recall_kit.server import create_app
    except ImportError:
        click.echo("Error: FastAPI and uvicorn are required for the server.")
        click.echo("Install them with: pip install 'recall-kit[mcp]'")
        return

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
