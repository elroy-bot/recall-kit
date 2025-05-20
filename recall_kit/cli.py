"""
Command-line interface for Recall Kit.

This module provides a CLI for interacting with Recall Kit, including commands
for creating, searching, and managing memories.
"""

import json
import logging
import sys
from typing import List, Optional

import click

from recall_kit import __version__
from recall_kit.storage import SQLiteBackend

from .core import RecallKit
from .repository.memory_store import MemoryStore
from .storage.base import Memory


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
    ctx.obj["recall"] = RecallKit.create(storage)


@cli.command()
@click.argument("text")
@click.option("--title", help="Title for the memory")
@click.option("--source", help="Source address for the memory")
@click.pass_context
def remember(
    ctx: click.Context, text: str, title: Optional[str], source: Optional[str]
):
    """Create a new memory."""
    memory_store: MemoryStore = ctx.obj["memory_store"]
    memory: Memory = memory_store.create_memory(
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
    memory_store: MemoryStore = ctx.obj["memory_store"]
    results: List[Memory] = memory_store.search(query, limit=limit)

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
@click.argument("path", type=click.Path(exists=True))
@click.option("--include", help="Glob pattern to include files")
@click.option("--exclude", help="Glob pattern to exclude files")
@click.option(
    "--recursive/--no-recursive", default=True, help="Recursively process directories"
)
@click.pass_context
def ingest(
    ctx: click.Context,
    path: str,
    include: Optional[str],
    exclude: Optional[str],
    recursive: bool,
):
    """Ingest documents from a directory."""
    import glob
    import os

    recall: RecallKit = ctx.obj["recall"]
    memory_store: MemoryStore = ctx.obj["memory_store"]

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
            memory = memory_store.create_memory(
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
