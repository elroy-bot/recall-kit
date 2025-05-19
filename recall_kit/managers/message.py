"""
Message management functionality for Recall Kit.

This module contains the MessageManager class for creating, retrieving, and managing messages and message sets.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from recall_kit.constants import ASSISTANT, CONTENT, ROLE, TOOL, USER
from recall_kit.models import Message, MessageSet


class MessageManager:
    """
    Manager class for message-related operations.

    This class provides methods for creating, retrieving, and managing messages and message sets.
    """

    def __init__(self, recall_kit):
        """
        Initialize a new MessageManager instance.

        Args:
            recall_kit: The RecallKit instance this manager belongs to
        """
        self.recall_kit = recall_kit
        self.storage = recall_kit.storage

    def create_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """
        Create a new message.

        Args:
            role: The role of the message sender (system, user, assistant, tool)
            content: The content of the message
            metadata: Additional metadata about the message
            user_id: ID of the user who owns this message (defaults to default user if not provided)
            tool_call_id: ID of the tool call this message is responding to (for tool messages)
            tool_calls: Tool calls made by this message (for assistant messages)

        Returns:
            The created Message object
        """
        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        message_args = {
            "role": role,
            CONTENT: content,
            "metadata": metadata or {},
            "user_id": user_id,
        }

        # Add tool-specific fields if provided
        if role == TOOL and tool_call_id:
            message_args["tool_call_id"] = tool_call_id

        if role == ASSISTANT and tool_calls:
            message_args["tool_calls"] = tool_calls

        message = Message(**message_args)

        # Store the message
        self.storage.store_message(message)

        return message

    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            The Message object if found, None otherwise
        """
        return self.storage.get_message(message_id)

    def get_all_messages(self) -> List[Message]:
        """
        Get all messages.

        Returns:
            List of all Message objects
        """
        return self.storage.get_all_messages()

    def create_message_set(
        self,
        message_ids: List[str],
        active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> MessageSet:
        """
        Create a new message set.

        Args:
            message_ids: List of message IDs in this set
            active: Whether this message set is active
            metadata: Additional metadata about the message set
            user_id: ID of the user who owns this message set (defaults to default user if not provided)

        Returns:
            The created MessageSet object
        """
        # If this is an active message set, deactivate all other message sets
        if active:
            self.storage.deactivate_all_message_sets()

        # Get default user_id if not provided
        if user_id is None:
            user_id = self.storage.get_default_user_id()

        message_set = MessageSet(
            message_ids=message_ids,
            active=active,
            metadata=metadata or {},
            user_id=user_id,
        )

        # Store the message set
        self.storage.store_message_set(message_set)

        return message_set

    def get_message_set(self, message_set_id: str) -> Optional[MessageSet]:
        """
        Get a message set by ID.

        Args:
            message_set_id: The ID of the message set to retrieve

        Returns:
            The MessageSet object if found, None otherwise
        """
        return self.storage.get_message_set(message_set_id)

    def get_active_message_set(self) -> Optional[MessageSet]:
        """
        Get the active message set.

        Returns:
            The active MessageSet object if found, None otherwise
        """
        return self.storage.get_active_message_set()

    def get_messages_in_set(self, message_set_id: str) -> List[Message]:
        """
        Get all messages in a message set.

        Args:
            message_set_id: The ID of the message set

        Returns:
            List of Message objects in the message set
        """
        return self.storage.get_messages_in_set(message_set_id)

    def deactivate_all_message_sets(self) -> None:
        """
        Deactivate all message sets.
        """
        self.storage.deactivate_all_message_sets()

    def store_conversation(
        self,
        messages: List[Dict[str, str]],
        response: Any,
        user_id: Optional[int] = None,
    ) -> MessageSet:
        """
        Store a conversation as a message set.

        Args:
            messages: List of messages in the conversation
            response: The response from the LLM
            user_id: ID of the user who owns this conversation

        Returns:
            The created MessageSet object
        """
        # Get the active message set
        active_message_set = self.get_active_message_set()

        # If there's only one user message and an active message set, add to it
        if len(messages) == 1 and messages[0].get(ROLE) == USER and active_message_set:
            # Create a new message for the user input
            user_message = self.create_message(
                role=USER,
                content=messages[0].get(CONTENT, ""),
                metadata={"type": "conversation"},
                user_id=user_id,
            )

            # Create a new message for the assistant response
            # Handle both object-style and dict-style responses
            if hasattr(response, "choices"):
                assistant_content = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                choice = response["choices"][0]
                if isinstance(choice, dict) and "message" in choice:
                    assistant_content = choice["message"].get(CONTENT, "")
                else:
                    return active_message_set
            else:
                return active_message_set

            assistant_message = self.create_message(
                role=ASSISTANT,
                content=assistant_content,
                metadata={"type": "conversation"},
                user_id=user_id,
            )

            # Update the message set with the new messages
            message_ids = active_message_set.message_ids + [
                user_message.id,
                assistant_message.id,
            ]

            # Create a new message set with the updated message IDs
            return self.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"type": "conversation"},
                user_id=user_id,
            )
        else:
            # Create new messages for each message in the conversation
            message_ids = []

            # If there's an active message set, check for duplicate messages
            existing_messages = []
            if active_message_set:
                existing_messages = self.get_messages_in_set(active_message_set.id)

            # Process each message
            for i, msg in enumerate(messages):
                role = msg.get(ROLE, "")
                content = msg.get(CONTENT, "")
                tool_call_id = msg.get("tool_call_id")
                tool_calls = msg.get("tool_calls")

                # Check if this message already exists in the active message set
                duplicate = False
                for existing_msg in existing_messages:
                    if existing_msg.role == role and existing_msg.content == content:
                        message_ids.append(existing_msg.id)
                        duplicate = True
                        break

                # If not a duplicate, create a new message
                if not duplicate:
                    # Handle tool messages properly
                    if (
                        role == TOOL
                        and i > 0
                        and messages[i - 1].get(ROLE) == ASSISTANT
                    ):
                        # If tool_call_id is not provided, generate one
                        if not tool_call_id:
                            import uuid

                            # Use a shorter prefix to ensure ID is under 40 characters
                            tool_call_id = f"c_{str(uuid.uuid4())}"

                        message = self.create_message(
                            role=role,
                            content=content,
                            metadata={"type": "conversation"},
                            user_id=user_id,
                            tool_call_id=tool_call_id,
                        )
                    elif role == ASSISTANT and tool_calls:
                        message = self.create_message(
                            role=role,
                            content=content,
                            metadata={"type": "conversation"},
                            user_id=user_id,
                            tool_calls=tool_calls,
                        )
                    else:
                        message = self.create_message(
                            role=role,
                            content=content,
                            metadata={"type": "conversation"},
                            user_id=user_id,
                        )
                    message_ids.append(message.id)

            # Create a message for the assistant response
            # Handle both object-style and dict-style responses
            if hasattr(response, "choices"):
                assistant_content = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                choice = response["choices"][0]
                if isinstance(choice, dict) and "message" in choice:
                    assistant_content = choice["message"].get(CONTENT, "")
                else:
                    # Skip adding assistant message if we can't extract content
                    return self.create_message_set(
                        message_ids=message_ids,
                        active=True,
                        metadata={"type": "conversation"},
                        user_id=user_id,
                    )
            else:
                # Skip adding assistant message if response format is unexpected
                return self.create_message_set(
                    message_ids=message_ids,
                    active=True,
                    metadata={"type": "conversation"},
                    user_id=user_id,
                )

            assistant_message = self.create_message(
                role=ASSISTANT,
                content=assistant_content,
                metadata={"type": "conversation"},
                user_id=user_id,
            )
            message_ids.append(assistant_message.id)

            # Create a new message set with the messages
            return self.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"type": "conversation"},
                user_id=user_id,
            )

    def compress_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        target_token_count: int = 4000,
        max_message_age: Optional[datetime.timedelta] = None,
    ) -> List[Dict[str, str]]:
        """
        Compress messages to fit within the context window, by summarizing earlier messages.

        Creates memories from compressed messages that are dropped from the context.
        Appends a summary of dropped messages to the earliest kept assistant message.
        Creates a new message set with the kept messages and marks the old one inactive.

        Args:
            messages: List of messages to compress
            model: Model name to use for token counting
            target_token_count: Target number of tokens to keep
            max_message_age: Maximum age of messages to keep (None means no limit)

        Returns:
           Compressed messages
        """
        import datetime
        from collections import deque

        from litellm import token_counter

        from recall_kit.constants import ASSISTANT, SYSTEM

        if not messages:
            return []

        # Find system message if it exists
        system_messages = [msg for msg in messages if msg.get(ROLE) == SYSTEM]
        non_system_messages = [msg for msg in messages if msg.get(ROLE) != SYSTEM]

        # If no system message, we'll just work with all messages
        if system_messages:
            system_message = system_messages[0]
            current_token_count = token_counter(
                model=model, text=system_message.get(CONTENT, "")
            )
        else:
            system_message = None
            current_token_count = 0

        kept_messages = deque()
        dropped_messages = []

        # Process messages in reverse order (newest first)
        for msg in reversed(non_system_messages):
            msg_content = msg.get(CONTENT, "")
            msg_role = msg.get(ROLE, "")

            # Calculate tokens for this message
            msg_token_count = token_counter(model=model, text=msg_content)

            # Check if we need to keep this message due to tool calls
            if (
                len(kept_messages) > 0
                and kept_messages[0].get(ROLE) == TOOL
                and msg_role == ASSISTANT
            ):
                # If the last message kept was a tool call, we must keep the corresponding assistant message
                kept_messages.appendleft(msg)
                current_token_count += msg_token_count
                continue

            # Check if we've exceeded our token budget
            if current_token_count + msg_token_count > target_token_count:
                # This message would put us over the limit
                dropped_messages.append(msg)
                continue

            # Check if the message is too old (if we have a max age)
            if max_message_age and "created_at" in msg:
                msg_created_at = msg.get("created_at")
                if isinstance(msg_created_at, str):
                    msg_created_at = datetime.datetime.fromisoformat(msg_created_at)

                if msg_created_at < datetime.datetime.now() - max_message_age:
                    dropped_messages.append(msg)
                    continue

            # If we get here, keep the message
            kept_messages.appendleft(msg)
            current_token_count += msg_token_count

        # Create memories from dropped messages
        if dropped_messages:
            # Create a consolidated memory from the dropped messages
            dropped_text = "\n".join(
                [
                    f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                    for msg in dropped_messages
                ]
            )

            # Create a memory from the dropped messages
            memory = self.recall_kit.memory_manager.create_memory(
                text=dropped_text,
                title=f"Compressed messages from conversation",
                metadata={"compressed": True, "message_count": len(dropped_messages)},
            )

            # Find the earliest kept assistant message to append the summary
            earliest_assistant_msg = None
            for msg in kept_messages:
                if msg.get(ROLE) == ASSISTANT:
                    earliest_assistant_msg = msg
                    break

            # If we found an assistant message, add a tool call and insert a tool results message
            if earliest_assistant_msg:
                # Create a new tool message with the summary
                summary_content = f"[Context: {len(dropped_messages)} earlier messages were summarized: {memory.title}]"

                # Create a tool message to insert after the earliest assistant message
                import uuid

                # Use a shorter prefix to ensure ID is under 40 characters
                tool_call_id = f"c_{str(uuid.uuid4())}"

                # Add tool_calls to the assistant message
                if "tool_calls" not in earliest_assistant_msg:
                    earliest_assistant_msg["tool_calls"] = [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "get_summary", "arguments": "{}"},
                        }
                    ]

                tool_message = {
                    "role": TOOL,
                    CONTENT: summary_content,
                    "metadata": {"type": "summary", "memory_id": memory.id},
                    "tool_call_id": tool_call_id,
                }

                # Find the index of the earliest assistant message
                earliest_assistant_idx = None
                for i, msg in enumerate(kept_messages):
                    if msg is earliest_assistant_msg:
                        earliest_assistant_idx = i
                        break

                # Insert the tool message after the earliest assistant message
                if earliest_assistant_idx is not None:
                    kept_messages.insert(earliest_assistant_idx + 1, tool_message)

        # Construct the final message list
        result = list(kept_messages)
        if system_message:
            result.insert(0, system_message)

        # Create a new message set with the kept messages and mark the old one inactive
        if result:
            # Store the messages in the database
            message_objects = []
            for i, msg in enumerate(result):
                role = msg.get(ROLE, "")
                tool_call_id = msg.get("tool_call_id")
                tool_calls = msg.get("tool_calls")

                # Create message with appropriate parameters based on role
                if role == TOOL and tool_call_id:
                    message = self.create_message(
                        role=role,
                        content=msg.get(CONTENT, ""),
                        metadata=msg.get("metadata", {}),
                        tool_call_id=tool_call_id,
                    )
                elif role == ASSISTANT and tool_calls:
                    message = self.create_message(
                        role=role,
                        content=msg.get(CONTENT, ""),
                        metadata=msg.get("metadata", {}),
                        tool_calls=tool_calls,
                    )
                else:
                    message = self.create_message(
                        role=role,
                        content=msg.get(CONTENT, ""),
                        metadata=msg.get("metadata", {}),
                    )
                message_objects.append(message)

            # Create a new message set
            message_ids = [msg.id for msg in message_objects]

            # If we have an existing message set, mark it inactive
            old_message_set = self.get_active_message_set()

            if old_message_set:
                old_message_set.active = False
                self.storage.store_message_set(old_message_set)

            # Create a new active message set
            self.create_message_set(
                message_ids=message_ids,
                active=True,
                metadata={"compressed": True},
            )

        return result
