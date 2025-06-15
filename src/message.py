from datetime import datetime
from typing import Any, Optional
from .constants import USER, ASSISTANT, SYSTEM, TOOL


def get_created_at(message) -> Optional[datetime]:
    """
    Extracts the created_at timestamp from a message.

    If the message is a MessageLike object, it returns its created_at property.
    If the message is a dictionary, it returns the 'created_at' key if it exists.
    """
    dt = _get_attr(message, 'created_at')

    if dt is not None:
        if isinstance(dt, datetime):
            return dt
        elif isinstance(dt, str):
            try:
                return datetime.fromisoformat(dt)
            except ValueError:
                pass


def get_role(message) -> str:
    """
    Extracts the role from a message.

    If the message is a MessageLike object, it returns its role property.
    If the message is a dictionary, it returns the 'role' key if it exists.
    """
    role = _get_attr(message, 'role')
    assert role is not None and role in [USER, ASSISTANT, SYSTEM, TOOL]
    return role

def get_content(message) -> str:
    content = _get_attr(message, 'content')
    if not content:
        return ""
    else:
        assert isinstance(content, str)
        return content




def _get_attr(message, attr: str) -> Optional[Any]:
    if hasattr(message, attr):
        return getattr(message, attr)
    elif hasattr(message, '__getitem__'):
        return message.get(attr, None)


