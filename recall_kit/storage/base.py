"""
Base storage interfaces and shared utilities for Recall Kit.

This module provides the base storage interfaces, shared utilities, and SQLModel table
definitions used by storage backends.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel

# Set up logging
logger = logging.getLogger(__name__)


# SQLModel classes for database tables
class UserTable(SQLModel, table=True):
    """SQLModel for the users table."""

    __tablename__ = "users"

    id: int = Field(primary_key=True, default=None)
    token: str = Field(unique=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class MemoryTable(SQLModel, table=True):
    """SQLModel for the memories table."""

    __tablename__ = "memories"

    id: str = Field(primary_key=True)
    text: str
    title: str
    source_address: Optional[str] = None
    parent_ids: Optional[str] = None  # JSON string of parent IDs
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata
    active: bool = Field(default=True)
    user_id: int


class EmbeddingTable(SQLModel, table=True):
    """SQLModel for the embeddings table."""

    __tablename__ = "embeddings"

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()))
    source_table: str  # The table name that this embedding is for
    source_id: str  # The ID of the record in the source table
    embedding: bytes = Field(sa_column=Column(LargeBinary))
    md5: str  # MD5 hash of the textual content that was embedded
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata


class MessageTable(SQLModel, table=True):
    """SQLModel for the messages table."""

    __tablename__ = "messages"

    id: str = Field(primary_key=True)
    role: str
    content: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata
    user_id: int


class MessageSetTable(SQLModel, table=True):
    """SQLModel for the message_sets table."""

    __tablename__ = "message_sets"

    id: str = Field(primary_key=True)
    message_ids: str  # JSON string of message IDs
    active: bool = Field(default=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata
    user_id: int


# Shared utility functions for storage backends
def parse_json_field(json_str: Optional[str]) -> Any:
    """Parse a JSON string field from the database."""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str}")
        return {}


def serialize_json_field(data: Any) -> Optional[str]:
    """Serialize data to a JSON string for storage."""
    if not data:
        return None
    return json.dumps(data)
