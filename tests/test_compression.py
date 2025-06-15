from typing import Optional
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from src.compression import ContextCompressor
from src.constants import SYSTEM, TOOL, ASSISTANT, USER
from src.message import get_content


class MockMessage:
    """Mock message class for testing"""
    def __init__(self, role: str, content: str, created_at: Optional[datetime] = None):
        self.role = role
        self.content = content
        self.created_at = created_at

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
        }


def test_empty_messages_list(model_name: str):
    """Test compression with empty messages list"""
    compressor = ContextCompressor.for_model(model_name, 100)
    kept, dropped = compressor.compress([])

    assert kept == []
    assert dropped == []


def test_system_message_always_kept(model_name):
    """Test that system messages are always kept at the beginning"""
    messages = [
        MockMessage(SYSTEM, "You are a helpful assistant"),
        MockMessage(USER, "Hello"),
        MockMessage(ASSISTANT, "Hi there!")
    ]

    compressor = ContextCompressor.for_model(model_name, 50)  # Low token limit
    kept, dropped = compressor.compress(messages)

    assert len(kept) >= 1
    assert kept[0].role == SYSTEM
    assert kept[0].content == "You are a helpful assistant"


def test_no_system_message(model_name):
    """Test compression when there's no system message"""
    messages = [
        MockMessage(USER, "Hello"),
        MockMessage(ASSISTANT, "Hi there!")
    ]

    compressor = ContextCompressor.for_model(model_name, 1000)
    kept, dropped = compressor.compress(messages)

    assert len(kept) == 2
    assert len(dropped) == 0
    assert kept[0].role == USER
    assert kept[1].role == ASSISTANT


def test_token_limit_compression(model_name):
    """Test that messages are dropped when token limit is exceeded"""
    messages = [
        MockMessage(USER, "This is a very long message that should consume many tokens" * 100),
        MockMessage(ASSISTANT, "This is another long response"),
        MockMessage(USER, "Short"),
        MockMessage(ASSISTANT, "Brief")
    ]

    compressor = ContextCompressor.for_model(model_name, 20)  # Very low token limit
    kept, dropped = compressor.compress(messages)

    # Should keep the most recent messages within token limit
    assert len(kept) > 0
    assert len(dropped) > 0

    # Most recent messages should be kept
    assert kept[-1].content == "Brief"


def test_age_based_filtering():
    """Test that old messages are dropped based on age"""
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)

    def mock_token_counter(text):
        return 10  # All messages have same token count

    messages = [
        MockMessage(USER, "Old message", old_time),
        MockMessage(ASSISTANT, "Old response", old_time),
        MockMessage(USER, "Recent message", recent_time),
        MockMessage(ASSISTANT, "Recent response", recent_time)
    ]

    compressor = ContextCompressor(
        mock_token_counter,
        1000,  # High token limit
        max_in_context_message_age=timedelta(hours=1)
    )
    kept, dropped = compressor.compress(messages)

    # Old messages should be dropped
    assert len(dropped) == 2
    assert all(msg.created_at == old_time for msg in dropped)

    # Recent messages should be kept
    assert len(kept) == 2
    assert all(msg.created_at == recent_time for msg in kept)


def test_tool_message_preservation(model_name):
    """Test that tool messages and their corresponding assistant messages are kept together"""
    messages = [
        MockMessage(USER, "What's the weather?"),
        MockMessage(ASSISTANT, "I'll check the weather for you"),
        MockMessage(TOOL, "Weather API result: Sunny, 75°F"),
        MockMessage(USER, "Thanks"),
        MockMessage(ASSISTANT, "You're welcome!")
    ]

    compressor = ContextCompressor.for_model(model_name, 30)  # Limited tokens
    kept, dropped = compressor.compress(messages)

    # If a tool message is kept, the assistant message before it should also be kept
    tool_indices = [i for i, msg in enumerate(kept) if msg.role == TOOL]
    for tool_idx in tool_indices:
        if tool_idx > 0:
            assert kept[tool_idx - 1].role == ASSISTANT


def test_chronological_order_maintained(model_name):
    """Test that chronological order is maintained in kept messages"""
    messages = [
        MockMessage(USER, "First message", datetime.now(timezone.utc) - timedelta(minutes=10)),
        MockMessage(ASSISTANT, "First response", datetime.now(timezone.utc) - timedelta(minutes=9)),
        MockMessage(USER, "Second message", datetime.now(timezone.utc) - timedelta(minutes=5)),
        MockMessage(ASSISTANT, "Second response", datetime.now(timezone.utc) - timedelta(minutes=4)),
        MockMessage(USER, "Third message", datetime.now(timezone.utc) - timedelta(minutes=1)),
        MockMessage(ASSISTANT, "Third response", datetime.now(timezone.utc) - timedelta(minutes=0))
    ]

    compressor = ContextCompressor.for_model(model_name, 1000)
    kept, dropped = compressor.compress(messages)

    # Check that kept messages maintain chronological order
    for i in range(1, len(kept)):
        prev_val = kept[i-1].created_at
        newer_val = kept[i].created_at
        assert prev_val is not None and newer_val is not None and prev_val <= newer_val


def test_system_message_with_compression(model_name):
    """Test system message handling with other message compression"""
    messages = [
        MockMessage(SYSTEM, "You are a helpful assistant"),
        MockMessage(USER, "Very long message that takes many tokens"),
        MockMessage(ASSISTANT, "Very long response that also takes many tokens" * 100),
        MockMessage(USER, "Short"),
        MockMessage(ASSISTANT, "Brief")
    ]

    compressor = ContextCompressor.for_model(model_name, 25)  # Low token limit
    kept, dropped = compressor.compress(messages)

    # System message should always be first in kept messages
    assert kept[0].role == SYSTEM
    assert kept[0].content == "You are a helpful assistant"

    # Some non-system messages should be dropped due to token limit
    assert len(dropped) > 0


def test_all_messages_within_limits(model_name):
    """Test when all messages fit within token and age limits"""
    messages = [
        MockMessage(USER, "Hello"),
        MockMessage(ASSISTANT, "Hi"),
        MockMessage(USER, "How are you?"),
        MockMessage(ASSISTANT, "I'm good!")
    ]

    compressor = ContextCompressor.for_model(model_name, 1000)  # High token limit
    kept, dropped = compressor.compress(messages)

    assert len(kept) == 4
    assert len(dropped) == 0
    assert kept == messages


def test_return_tuple_structure(model_name):
    """Test that compress returns a proper tuple structure"""
    messages = [MockMessage(USER, "Test message")]

    compressor = ContextCompressor.for_model(model_name, 1000)
    result = compressor.compress(messages)

    assert isinstance(result, tuple)
    assert len(result) == 2
    kept, dropped = result
    assert isinstance(kept, list)
    assert isinstance(dropped, list)


def test_token_limit_compression_dict(model_name):
    """Test that messages are dropped when token limit is exceeded"""
    messages = [
        MockMessage(USER, "This is a very long message that should consume many tokens" * 100).to_dict(),
        MockMessage(ASSISTANT, "This is another long response").to_dict(),
        MockMessage(USER, "Short").to_dict(),
        MockMessage(ASSISTANT, "Brief").to_dict()
    ]

    compressor = ContextCompressor.for_model(model_name, 20)  # Very low token limit
    kept, dropped = compressor.compress(messages)

    # Should keep the most recent messages within token limit
    assert len(kept) > 0
    assert len(dropped) > 0

    # Most recent messages should be kept
    assert get_content(kept[-1]) == "Brief"
