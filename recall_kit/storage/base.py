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

from litellm import AllMessageValues
from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel

# Set up logging
logger = logging.getLogger(__name__)


# SQLModel classes for database tables
class User(SQLModel, table=True):
    """SQLModel for the users table."""

    id: Optional[int] = Field(default=None, primary_key=True)
    token: str = Field(unique=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class Memory(SQLModel, table=True):
    """SQLModel for the memories table."""

    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    title: str
    source_address: Optional[str] = None
    parent_ids: Optional[str] = None  # JSON string of parent IDs
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata
    active: bool = Field(default=True)
    user_id: int


class Embedding(SQLModel, table=True):
    """SQLModel for the embeddings table."""

    id: Optional[int] = Field(default=None, primary_key=True)
    source_table: str  # The table name that this embedding is for
    source_id: str  # The ID of the record in the source table
    embedding: bytes = Field(sa_column=Column(LargeBinary))
    md5: str  # MD5 hash of the textual content that was embedded
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    text: str = Field(default="")

    def set_message_value(self, message_values: AllMessageValues) -> None:
        """Serialize AllMessageValues to the text column"""
        self.text = json.dumps(message_values)

    def get_message_value(self) -> AllMessageValues:
        """Deserialize text column to AllMessageValues"""
        return json.loads(self.text)


class MessageSet(SQLModel, table=True):
    """SQLModel for the message_sets table."""

    id: Optional[int] = Field(default=None, primary_key=True)
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
