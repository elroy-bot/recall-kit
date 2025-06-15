from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple, TypeVar

from .constants import SYSTEM, TOOL
from .message import get_content, get_role, get_created_at

T = TypeVar("T")


class ContextCompressor:
    def __init__(
        self,
        token_counter_fn: Callable[[str], int],
        context_refresh_target_tokens: int,
        max_in_context_message_age: Optional[timedelta] = None,
    ):
        self.token_counter_fn = token_counter_fn
        self.context_refresh_target_tokens = context_refresh_target_tokens
        self.max_in_context_message_age = max_in_context_message_age

    @classmethod
    def for_model(cls, model_name: str, context_refresh_target_tokens: int, max_in_context_message_age: Optional[timedelta] = None) -> "ContextCompressor":
        from litellm.utils import token_counter
        return cls(
            token_counter_fn=lambda text: token_counter(model=model_name, text=text),
            context_refresh_target_tokens=context_refresh_target_tokens,
            max_in_context_message_age=max_in_context_message_age,
        )

    def compress(self, messages: List[T]) -> Tuple[List[T], List[T]]:
        """
        Compresses the context messages by summarizing old messages while keeping new messages intact.

        Returns a tuple of two lists. The first list contains the messages that should be kept in context, the second list contains the messages that were dropped.
        """
        if not messages:
            return ([],[])

        kept_messages = deque()

        # keep system message, if it exists
        if get_role(messages[0]) == SYSTEM:
            system_message = messages[0]
            messages = messages[1:]
            current_token_count = self.token_counter_fn(get_content(system_message))
        else:
            system_message = None
            current_token_count = 0

        # we keep the most current messages that are fresh enough to be relevant
        curr_idx = len(messages) - 1

        while curr_idx >= 0:
            msg = messages[curr_idx]
            candidate_message_count = self.token_counter_fn(get_content(msg))
            msg_created_at = get_created_at(msg)


            # if the last message kept was a tool call, we must keep the corresponding assistant message that came before it.
            if len(kept_messages) > 0 and get_role(kept_messages[0]) == TOOL:
                kept_messages.appendleft(msg)
                current_token_count += self.token_counter_fn(get_content(msg))

            # If adding this message would exceed the token limit, we drop it and all previous messages
            elif candidate_message_count + current_token_count > self.context_refresh_target_tokens:
                break

            # If the message is timestamped and too old, drop it and all previous messages
            elif msg_created_at and self.max_in_context_message_age and msg_created_at < datetime.now(timezone.utc) - self.max_in_context_message_age:
                break

            # we made it! keep the message.
            else:
                kept_messages.appendleft(msg)
                current_token_count += candidate_message_count
            curr_idx -= 1

        system_message_list = [system_message] if system_message else []
        return (system_message_list + list(kept_messages), list(messages[:curr_idx + 1]))


