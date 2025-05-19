"""
Chat management functionality for Recall Kit.

This module contains the ChatManager class for chat completions and augmentation.
"""

from __future__ import annotations

from typing import Any, Dict

from litellm import ContextWindowExceededError


class ChatManager:
    """
    Manager class for chat-related operations.

    This class provides methods for chat completions and augmentation.
    """

    def __init__(self, recall_kit):
        """
        Initialize a new ChatManager instance.

        Args:
            recall_kit: The RecallKit instance this manager belongs to
        """
        self.recall_kit = recall_kit
        self.storage = recall_kit.storage
        self.completion_fn = recall_kit.completion_fn
        self.augment_fn = recall_kit.augment_fn

    def augment_chat_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat completion request with memory augmentation.

        Args:
            request: The chat completion request

        Returns:
            Augmented chat completion request
        """
        # Extract the query from the last user message
        memories = self.recall_kit.memory_manager.get_relevant_memories(request)
        # Augment the request with memories
        augmented_request = self.augment_fn(memories, request)

        return augmented_request

    def completion(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate an OpenAI compatible chat completion with memory augmentation.

        Args:
            **kwargs: Arguments to pass to the chat completion API
                user: Optional[str] - A unique identifier representing the end-user

        Returns:
            Chat completion response
        """
        from recall_kit.constants import MESSAGES, MODEL, USER, USER_ID

        # Process the request with memory augmentation
        augmented_request = self.augment_chat_request(kwargs)

        # Extract parameters for the completion function
        model = augmented_request.get(MODEL, None)
        if not model:
            raise ValueError("No model specified for chat completion")

        messages = augmented_request.pop(MESSAGES, [])
        max_tokens = augmented_request.pop("max_tokens", None)
        temperature = augmented_request.pop("temperature", None)

        # Extract user parameter if provided
        user_token = augmented_request.pop(USER, "default")

        # Get or create user ID from token
        user_id = self.storage.get_user_by_token(user_token)
        if user_id is None:
            user_id = self.storage.create_user(user_token)

        # Add any remaining kwargs to additional args
        additional_args = augmented_request
        # Add user token back to additional args for the completion function
        additional_args[USER] = user_token

        # Generate the completion using the completion function
        try:
            response = self.completion_fn(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                additional_args=additional_args,
            )
        except ContextWindowExceededError:
            # If context window is exceeded, compress messages and try again
            kwargs[MESSAGES] = self.recall_kit.message_manager.compress_messages(
                model=model, messages=messages
            )
            return self.completion(**kwargs)

        # Store the conversation as a memory and as messages
        self._store_conversation_memory(
            {MESSAGES: messages, MODEL: model, USER_ID: user_id}, response
        )
        self.recall_kit.message_manager.store_conversation(messages, response, user_id)

        return response

    def _store_conversation_memory(
        self, request: Dict[str, Any], response: Any
    ) -> None:
        """
        Store a conversation as a memory.

        Args:
            request: The chat completion request
            response: The chat completion response
        """
        from recall_kit.constants import CONTENT, ROLE, USER

        messages = request.get("messages", [])
        user_messages = [m for m in messages if m.get(ROLE) == USER]
        user_id = request.get("user_id")

        if not user_messages:
            return

        query = user_messages[-1].get(CONTENT, "")

        # Handle both object-style and dict-style responses
        if hasattr(response, "choices"):
            assistant_message = response.choices[0].message.content
        elif isinstance(response, dict) and "choices" in response:
            choice = response["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                assistant_message = choice["message"].get(CONTENT, "")
            else:
                return
        else:
            return

        conversation_text = f"User: {query}\nAssistant: {assistant_message}"

        self.recall_kit.memory_manager.create_memory(
            text=conversation_text,
            title=query[:50] + "..." if len(query) > 50 else query,
            metadata={"type": "conversation"},
            user_id=user_id,
        )
