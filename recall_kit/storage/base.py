"""
Base storage interfaces and shared utilities for Recall Kit.

This module provides the base storage interfaces, shared utilities, and SQLModel table
definitions used by storage backends.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

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
    _parent_ids: Optional[str] = None  # JSON string of parent IDs
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    _meta_data: Optional[str] = None  # JSON string of metadata
    active: bool = Field(default=True)
    user_id: int

    @property
    def meta_data(self) -> Dict[str, Any]:
        """Parse the meta_data field into a dictionary."""
        if self._meta_data:
            try:
                return json.loads(self._meta_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse meta_data: {self._meta_data}")
        return {}

    @meta_data.setter
    def meta_data(self, value: Dict[str, Any]) -> None:
        """Set the meta_data field from a dictionary."""
        assert isinstance(value, dict), "meta_data must be a dictionary"
        self._meta_data = json.dumps(value)

    @property
    def parent_ids(self) -> List[int]:
        """Parse the parent_ids field into a list of integers."""
        if self._parent_ids:
            try:
                return json.loads(self._parent_ids)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse parent_ids: {self._parent_ids}")
        return []

    @parent_ids.setter
    def parent_ids(self, value: List[int]) -> None:
        """Set the parent IDs from a list of integers."""
        self._parent_ids = json.dumps(value)

    @property
    def relevance(self) -> Optional[float]:
        return getattr(self, "_relevance")

    @relevance.setter
    def relevance(self, val: float) -> None:
        setattr(self, "_relevance", val)


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
    _data: str = Field(default="")

    @property
    def data(self) -> AllMessageValues:
        """Deserialize text column to AllMessageValues"""
        return json.loads(self._data)

    @data.setter
    def data(self, value: AllMessageValues) -> None:
        """Serialize AllMessageValues to the text column"""
        assert isinstance(value, dict), "data must be a dictionary"
        self._data = json.dumps(value)


class MessageSet(SQLModel, table=True):
    """SQLModel for the message_sets table."""

    id: Optional[int] = Field(default=None, primary_key=True)
    _message_ids: str  # JSON string of message IDs
    active: bool = Field(default=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    meta_data: Optional[str] = None  # JSON string of metadata
    user_id: int

    @property
    def message_ids(self) -> List[int]:
        if not self._message_ids:
            return []
        else:
            return json.loads(self._message_ids)


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
