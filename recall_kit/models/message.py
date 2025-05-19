"""
Message-related data models for Recall Kit.

This module contains the Message and MessageSet classes used throughout Recall Kit.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..constants import ASSISTANT, SYSTEM, TOOL, USER


class Message(BaseModel):
    """
    A message in a conversation.

    This represents a single message in a conversation, which can be from a user,
    assistant, or system.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    role: str = Field(
        ...,
        description="The role of the message sender (system, user, assistant, tool)",
    )
    content: str = Field(..., description="The content of the message")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the message was created",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the message"
    )
    user_id: int = Field(..., description="ID of the user who owns this message")
    tool_call_id: Optional[str] = Field(
        None,
        description="ID of the tool call this message is responding to (for tool messages)",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls made by this message (for assistant messages)"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that the role is one of the allowed values."""
        allowed_roles = {SYSTEM, USER, ASSISTANT, TOOL}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        result = self.model_dump()
        # Remove None values for tool_call_id and tool_calls
        if result.get("tool_call_id") is None:
            result.pop("tool_call_id", None)
        if result.get("tool_calls") is None:
            result.pop("tool_calls", None)
        return result


class MessageSet(BaseModel):
    """
    A set of messages in a conversation.

    This represents a collection of messages in a conversation, which can be
    used to track the history of a conversation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message set",
    )
    message_ids: List[str] = Field(
        default_factory=list, description="IDs of messages in this set"
    )
    active: bool = Field(default=True, description="Whether this message set is active")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the message set was created",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the message set"
    )
    user_id: int = Field(..., description="ID of the user who owns this message set")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message set to a dictionary."""
        return self.model_dump()
