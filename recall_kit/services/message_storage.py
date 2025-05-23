import json
from litellm import AllMessageValues, ChatCompletionRequest

from ..models.sql_models import MessageSet

from ..constants import DEFAULT, MESSAGES, USER, USER_ID
from ..protocols.base import StorageBackendProtocol


class MessageStorageService:
    def __init__(self, storage: StorageBackendProtocol):
        """
        Initialize a new MessageStorageService instance.

        Args:
            storage: The storage backend to use for message management
        """
        self.storage = storage

    def get_stored_messages(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Get a message set.

        Args:
            message_set: The message set to retrieve

        Returns:
            The message set
        """





        # retrieve active message set for user
        # if there is no active message set, create one with the messages in the request.
        # if there is an active message set, retrieve it. Examine the messages in the request, and see if they match the messages in the active message set, starting from the oldest message in the incoming request.
        # if an overlap is found, append the active message set to the messages from ithe incoming request.

        # keep track of which messages were already persisted, and which have not been.
        # persist the new messages, create a new message set, update the request, and return it.

        # if there are mutliple messages in the request, examine each to


        active_message_set = self.storage.get_active_message_set()

        user_id= self.storage.get_user_by_token(request.get(USER)) or self.storage.get_default_user_id()

        if not active_message_set:
            message_ids = self.storage.store_messages(request[MESSAGES])
            message_set = self.storage.store_message_set(MessageSet(user_id=user_id, message_ids_str=json.dumps(message_ids)),)
            return request
        else:
            # TODO: COMPLETE THE FUNCTION



