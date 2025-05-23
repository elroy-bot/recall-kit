"""
Command-line interface for Recall Kit.

This module provides a CLI for interacting with Recall Kit, including commands
for creating, searching, and managing memories.
"""

import logging
import sys
from typing import Optional

import click

from recall_kit import __version__
from recall_kit.storage import SQLiteBackend

from .core import RecallKit


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
    ctx.obj["recall"] = RecallKit(embedding_model=embedding_model, storage=storage)


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
