# RecallKit

RecallKit is a lightweight memory integration toolkit for Large Language Models (LLMs). It provides utilities to manage and optimize the context window of LLM conversations.

## Current Features

### Message Compression

The core functionality currently implemented is context compression for LLM conversations. When working with LLMs, the context window (the amount of text the model can consider at once) is limited. As conversations grow longer, they may exceed this limit, requiring some messages to be dropped or summarized.

RecallKit's `ContextCompressor` intelligently manages this process by:

- Preserving system messages (instructions to the LLM)
- Prioritizing recent messages over older ones
- Respecting token limits for different models
- Optionally filtering out messages older than a specified age
- Maintaining the relationship between tool calls and their corresponding messages

#### Example Usage

```python
from recallkit.compression import ContextCompressor
from datetime import timedelta

# Create a compressor for a specific model with a token target
compressor = ContextCompressor.for_model(
    model_name="gpt-4o-mini",
    context_refresh_target_tokens=4000,
    max_in_context_message_age=timedelta(hours=1)  # Optional
)

# Compress a list of messages
kept_messages, dropped_messages = compressor.compress(conversation_messages)

# Use kept_messages in your next LLM call
```

The compressor works with both object-style messages (with attributes) and dictionary-style messages (compatible with most LLM APIs).

## Future Extensions

RecallKit is designed to be extended with additional memory management features for LLMs. Future versions may include:

- Conversation summarization
- Long-term memory storage and retrieval
- Semantic search across conversation history
- Integration with vector databases
- Customizable compression strategies

## Installation

```bash
pip install recallkit
```

## Requirements

- Python 3.10+
- litellm (for token counting)

## License

MIT
