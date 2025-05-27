import json
from typing import List

from litellm import AllMessageValues, ChatCompletionRequest

from ..constants import MESSAGES, USER
from ..models.sql_models import MessageSet
from ..protocols.base import StorageBackendProtocol


class MessageStorageService:
    def __init__(self, storage: StorageBackendProtocol, user_id: int):
        """
        Initialize a new MessageStorageService instance.

        Args:
            storage: The storage backend to use for message management
        """
        self.storage = storage
        self.user_id = user_id

    def set_conversation_messages(self, messages: List[AllMessageValues]) -> int:
        return self.replace_active_message_set(self.storage.store_messages(messages))

    def replace_active_message_set(self, message_ids: List[int]) -> int:
        self.storage.deactivate_all_message_sets()
        return self.storage.store_message_set(
            MessageSet(
                user_id=self.user_id,
                message_ids_str=json.dumps(message_ids),
                active=True,
            )
        )

    def create_inactive_message_set(self, messages: List[AllMessageValues]) -> int:
        """
        Create a new inactive message set.

        Args:
            message_ids: List of message IDs to include in the set

        Returns:
            The ID of the newly created message set
        """
        message_ids = self.storage.store_messages(messages)

        return self.storage.store_message_set(
            MessageSet(
                user_id=self.user_id,
                message_ids_str=json.dumps(message_ids),
                active=False,
            )
        )

    def get_stored_messages(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        """
        Get a message set.

        Args:
            message_set: The message set to retrieve

        Returns:
            The message set
        """

        active_message_set = self.storage.get_active_message_set()

        user_id = (
            self.storage.get_user_by_token(request.get(USER))
            or self.storage.get_default_user_id()
        )

        if not active_message_set:
            message_ids = self.storage.store_messages(request[MESSAGES])
            self.replace_active_message_set(message_ids=message_ids)
            return request
        else:
            # Get the messages from the active message set
            # We already checked that active_message_set is not None
            assert (
                active_message_set is not None and active_message_set.id is not None
            ), "Active message set or its ID is None"
            active_messages = self.storage.get_messages_in_set(active_message_set.id)

            # Get the incoming messages
            incoming_messages = request[MESSAGES]

            # Find overlap between active messages and incoming messages
            # We'll check if the oldest incoming messages match with the most recent active messages
            overlap_found = False
            overlap_index = 0

            # Try to find an overlap by comparing messages
            for i in range(len(incoming_messages)):
                if i >= len(active_messages):
                    break

                # Compare messages starting from the beginning of incoming messages
                # with the end of active messages (most recent ones)
                if json.dumps(incoming_messages[i]) == json.dumps(
                    active_messages[-(len(incoming_messages) - i)]
                ):
                    overlap_found = True
                    overlap_index = i
                    break

            if overlap_found:
                # If overlap is found, we need to merge the messages
                # Keep the messages that were already in the active set
                # and add only the new messages from the incoming request
                message_ids = active_message_set.message_ids

                # Store only the new messages (those after the overlap)
                new_messages = incoming_messages[overlap_index + 1 :]
                if new_messages:
                    new_message_ids = self.storage.store_messages(new_messages)
                    message_ids.extend(new_message_ids)

                self.replace_active_message_set(message_ids=message_ids)

                # Update the request with all messages
                all_messages = active_messages + new_messages
                request[MESSAGES] = all_messages
            else:
                # No overlap found, store all incoming messages as a new set
                message_ids = self.storage.store_messages(incoming_messages)

                # Create a new active message set
                self.replace_active_message_set(message_ids)

            return request
