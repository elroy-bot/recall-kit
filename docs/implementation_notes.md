# Implementation notes

This document outlines possible code interfaces for the Recall Kit package, focusing on integration methods, plugin systems, and customizable components.


## LLM available Functions
- `search_memories`
- `get_source_data`
- `create_memory`


## Customizable Components

Recall Kit provides several customizable components that can be tailored to specific needs:

### Storage Options

```python
from recall_kit import RecallKit

# Default SQLite storage
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()  # Uses default path
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

# Create RecallKit with the service instances
recall = RecallKit(storage=storage, embedding_service=embedding_service)

# PostgreSQL storage
from recall_kit.storage import PostgresBackend

# Create PostgreSQL storage backend
postgres_storage = PostgresBackend("postgresql://user:password@localhost:5432/memories")
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

# Create RecallKit with PostgreSQL storage
recall = RecallKit(storage=postgres_storage, embedding_service=embedding_service)
```

### Vector Store Options

```python
from recall_kit import RecallKit

# Default SQLite vector store (sqlite-vec)
# This example shows how to use SQLite for both storage and vector storage
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

storage = SQLiteBackend()  # Uses sqlite-vec by default
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(storage=storage, embedding_service=embedding_service)

# PostgreSQL with pgvector
from recall_kit.storage import PostgresBackend

# Create PostgreSQL storage with pgvector support
postgres_storage = PostgresBackend("postgresql://user:password@localhost:5432/memories")
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(storage=postgres_storage, embedding_service=embedding_service)
```

### Memory Source

```python
from recall_kit import MemorySource

# Create a memory source with required fields
source = MemorySource(
    text="The meeting with the design team is scheduled for May 15th at 2 PM.",
    title="Design Team Meeting",
    address="conversation:2023-05-10:message:42"  # Unique identifier for the source
)

# Convert to memory
memory = source.to_memory()

# Add to recall-kit
recall.add_memory(memory)
```

### Core Memory Functions

```python
from recall_kit import RecallKit
from openai.types.chat import ChatCompletionRequest

# Custom retrieve function
def my_retrieve(query_text: str, request: ChatCompletionRequest):
    # Custom semantic search logic
    return [...]  # List of memory objects

# Custom filter function
def my_filter(memory, request: ChatCompletionRequest):
    # Custom filtering logic
    return True if memory.relevance > 0.8 else False

# Custom rerank function
def my_rerank(memories, request: ChatCompletionRequest):
    # Custom reranking logic
    return sorted(memories, key=lambda m: m.recency * m.relevance, reverse=True)

# Custom augment function
def my_augment(memories, request: ChatCompletionRequest):
    # Custom logic to incorporate memories into the request
    augmented_request = request.copy()
    # Modify the request to include memories
    return augmented_request

# Initialize RecallKit with custom functions
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

# Create RecallKit with custom functions
recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service,
    retrieve_fn=my_retrieve,
    filter_fn=my_filter,      # Optional
    rerank_fn=my_rerank,      # Optional
    augment_fn=my_augment
)
```

### General thoughts

The tricky tradeoff to balance is giving enough customizability, while not requiring users to imeplement complicated functions that overwrite many parameters.

For example, the vector store should be configured in one place, and adding a MemorySource class should not require passing in all configuration for the vector store

## Interfaces

## CLI Integration

### Standalone CLI

```bash
# Create a memory
recall remember "The meeting with the design team is scheduled for May 15th at 2 PM."

# Search memories
recall search "When is the design team meeting?"

# Interactive chat with memory
recall chat

# Ingest documents
recall ingest ./project_docs --include "*.md" --exclude "README.md"

# Export memories
recall export --format json --output memories.json

# View memory statistics
recall stats
```

### LLM CLI Extension

```python
# Implementation of LLM extension
from llm import hookimpl
import click
from recall_kit import RecallKit

from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

storage = SQLiteBackend()
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")
recall = RecallKit(storage=storage, embedding_service=embedding_service)

@hookimpl
def register_commands(cli):
    @cli.group(name="recall")
    def recall_group():
        "Commands for managing recall-kit memories"
        pass

    @recall_group.command(name="remember")
    @click.argument("text")
    def remember(text):
        "Create a new memory"
        memory = recall.create_memory(text)
        click.echo(f"Memory created: {memory.id}")

    @recall_group.command(name="search")
    @click.argument("query")
    @click.option("--limit", default=5, help="Maximum number of results")
    def search(query, limit):
        "Search memories"
        results = recall.search(query, limit=limit)
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.text} (score: {result.score:.2f})")

@hookimpl
def register_models(register):
    register(RecallKitModel(), aliases=("recall",))

class RecallKitModel(llm.Model):
    model_id = "recall-kit"

    def execute(self, prompt, stream, response, conversation):
        # Retrieve relevant memories
        memories = recall.search(prompt, limit=3)

        # Add memories to the context
        context = "Relevant memories:\n"
        for memory in memories:
            context += f"- {memory.text}\n"

        # Forward to the underlying model with memory context
        underlying_model = llm.get_model("gpt-3.5-turbo")
        augmented_prompt = f"{context}\n\nUser query: {prompt}"

        yield from underlying_model.execute(
            augmented_prompt, stream, response, conversation
        )
```

Usage:
```bash
# Use the recall model
llm -m recall "What was the date of our design team meeting?"

# Use recall commands
llm recall remember "The project deadline is June 30th."
llm recall search "deadline"
```

### Smolagents Integration

Recall Kit offers two complementary approaches for integrating with [smolagents](https://github.com/huggingface/smolagents), each with distinct advantages for different use cases. You can implement either approach or combine them based on your specific requirements.

#### Approach 1: Recall Kit as Smolagents Tools

This approach provides explicit memory tools that the agent can use when needed, giving fine-grained control over memory operations.

##### Implementation

```python
from smolagents import Agent
from recall_kit.smolagents import RecallMemoryTool, CreateMemoryTool
from recall_kit import RecallKit

# Initialize Recall Kit with your preferred configuration
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()  # For SQLite storage
# Or use PostgresBackend for PostgreSQL storage
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service
)

# Create memory tools with the recall instance
recall_tool = RecallMemoryTool(recall_kit=recall)
create_tool = CreateMemoryTool(recall_kit=recall)

# Define the agent with Recall Kit tools
agent = Agent(
    tools=[recall_tool, create_tool],
    system_prompt="""You are an assistant with memory capabilities.
    Use the RecallMemoryTool to retrieve relevant memories when the user asks about past information.
    Use the CreateMemoryTool to store important information that might be needed later.

    Guidelines for using memory tools:
    1. Create memories for important facts about the user (preferences, goals, personal details)
    2. Create memories for key decisions or conclusions reached during the conversation
    3. Recall memories when answering questions about past interactions
    4. Don't create memories for trivial or temporary information"""
)

# Use the agent
response = agent.run("What do you remember about our previous conversations?")
print(response)

# Create a new memory
response = agent.run("Remember that I prefer to be called Alex instead of Alexander.")
print(response)

# Advanced: Configure memory tools with custom parameters
recall_tool = RecallMemoryTool(
    recall_kit=recall,
    max_results=5,
    relevance_threshold=0.75,
    include_sources=True
)

create_tool = CreateMemoryTool(
    recall_kit=recall,
    auto_consolidate=True,
    importance_threshold=0.6
)
```

##### Tool Definitions

The Recall Kit tools for smolagents provide the following functionality:

**RecallMemoryTool**
- **Purpose**: Retrieve memories relevant to a query
- **Parameters**:
  - `query` (required): The text to search for in memories
  - `max_results` (optional): Maximum number of memories to return (default: 3)
  - `include_sources` (optional): Whether to include source documents (default: False)
- **Returns**: List of relevant memories with optional source information

**CreateMemoryTool**
- **Purpose**: Store new memories
- **Parameters**:
  - `text` (required): The memory text to store
  - `title` (optional): A title for the memory (default: auto-generated)
  - `importance` (optional): Importance score from 0-1 (default: 0.5)
- **Returns**: Confirmation with the memory ID

##### Use Cases

This approach is ideal for:
- Chatbots that need selective memory operations
- Applications where memory usage should be transparent to the user
- Scenarios where memory operations should be explicitly logged
- Agents that need to combine memory with other specialized tools

#### Approach 2: Recall Kit as a Custom AgentMemory

This approach replaces the default memory system in smolagents with Recall Kit, providing automatic memory management without explicit tool calls.

##### Implementation

```python
from smolagents import Agent
from recall_kit.smolagents import RecallKitAgentMemory
from recall_kit import RecallKit

# Initialize Recall Kit with your preferred configuration
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()  # For SQLite storage
# Or use PostgresBackend for PostgreSQL storage
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service
)

# Create an agent with standard configuration
agent = Agent(
    tools=[],  # Can still include other tools
    system_prompt="""You are an assistant with perfect recall.
    Relevant memories will be provided to you automatically.
    Focus on providing helpful responses based on your knowledge and the context provided."""
)

# Replace the default AgentMemory with RecallKitAgentMemory
agent.memory = RecallKitAgentMemory(
    recall_kit=recall,
    system_prompt=agent.system_prompt,
    auto_consolidate=True,
    memory_prefix="Relevant memories from previous conversations:",
    max_memories=5,
    relevance_threshold=0.7,
    embedding_model="text-embedding-3-small"
)

# Use the agent
response = agent.run("Hi, I'm starting a new project on renewable energy.")
print(response)

# Later in the conversation
response = agent.run("What was I working on again?")
print(response)

# Explicitly add a memory if needed
agent.memory.add_memory("User is interested in solar panel efficiency improvements.")

# Advanced: Configure memory retrieval behavior
agent.memory.configure(
    auto_memory_capture=True,  # Automatically create memories from conversations
    consolidation_interval=10,  # Consolidate memories every 10 new memories
    memory_decay=0.05,         # Apply slight decay to older memories
    include_sources=False      # Don't include source documents by default
)
```

##### Memory Integration Process

When using RecallKitAgentMemory, the following process occurs automatically:

1. **Memory Capture**: Conversations are automatically processed to extract important information
2. **Memory Retrieval**: Before each agent run, relevant memories are retrieved based on the current query
3. **Context Augmentation**: Retrieved memories are added to the system prompt or as context messages
4. **Memory Consolidation**: Similar memories are periodically consolidated to maintain a concise memory store

##### Use Cases

This approach is ideal for:
- Long-running assistants that need seamless memory integration
- Applications where memory operations should be invisible to the user
- Scenarios requiring automatic memory management without explicit prompting
- Agents focused on natural conversation rather than tool use

#### Combining Both Approaches

For advanced use cases, you can combine both approaches to get the best of both worlds:

```python
from smolagents import Agent
from recall_kit.smolagents import RecallKitAgentMemory, RecallMemoryTool, CreateMemoryTool
from recall_kit import RecallKit

# Initialize Recall Kit
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service
)

# Create memory tools for explicit operations
recall_tool = RecallMemoryTool(recall_kit=recall)
create_tool = CreateMemoryTool(recall_kit=recall)

# Create an agent with both automatic memory and explicit tools
agent = Agent(
    tools=[recall_tool, create_tool],
    system_prompt="""You are an assistant with both automatic and manual memory capabilities.
    Relevant memories will be provided to you automatically.
    Use the RecallMemoryTool for specific memory searches beyond automatic retrieval.
    Use the CreateMemoryTool to explicitly store critical information."""
)

# Set up automatic memory
agent.memory = RecallKitAgentMemory(
    recall_kit=recall,  # Use the same recall instance for both
    system_prompt=agent.system_prompt,
    auto_consolidate=True
)

# The agent now has both automatic memory retrieval and explicit memory tools
response = agent.run("Tell me about our last conversation and also remember that I prefer dark mode for all applications.")
```

This combined approach allows for:
- Automatic memory retrieval for general context
- Explicit memory operations when more control is needed
- Unified memory store across both automatic and manual operations

**Pros and Cons:**

**Approach 1: Recall Kit as Smolagents Tools**
- Pros:
  - Direct control over memory operations
  - Clear visibility into memory actions in the agent's reasoning
  - Can be combined with other tools easily
  - Explicit memory operations are logged in the conversation
- Cons:
  - Requires explicit tool calls in prompts
  - May consume more tokens in the context window
  - Memory management logic must be handled in prompts
  - Requires more complex prompt engineering

**Approach 2: Recall Kit as a Custom AgentMemory**
- Pros:
  - Integrates with smolagents' existing memory architecture
  - Automatic memory retrieval and integration
  - Cleaner agent prompts without explicit memory management
  - More natural conversation flow without tool interruptions
- Cons:
  - Requires replacing the default memory implementation
  - May require more configuration for optimal performance
  - Potentially more complex to debug memory-related issues
  - Less control over when memories are created or retrieved

#### Implementation Considerations

When implementing either approach, consider the following:

1. **Storage Configuration**: Both approaches can use the same underlying RecallKit instance, allowing you to choose SQLite for development and PostgreSQL for production.

2. **Embedding Models**: Select an appropriate embedding model based on your requirements for accuracy vs. performance.

3. **Memory Consolidation**: Enable auto_consolidate to maintain a concise memory store, especially for long-running agents.

4. **Context Window Management**: Be mindful of how many memories are included in the context to avoid exceeding token limits.

5. **Error Handling**: Implement proper error handling for memory operations, especially when using explicit tools.

6. **Testing**: Test your memory implementation with various conversation patterns to ensure it behaves as expected.

### Plugin System

#### Approach 1: Hook-based Plugin System (Similar to LLM)

```python
from recall_kit import hookimpl, RecallKit

# Create a custom memory processor plugin
class CustomMemoryProcessor:
    def process_memory(self, memory_text):
        # Custom processing logic
        return f"Processed: {memory_text}"

# Register the plugin using the hookimpl decorator
@hookimpl
def register_memory_processors(register):
    register(CustomMemoryProcessor())

# Create a custom embedding model
class CustomEmbeddingModel:
    model_id = "custom-embeddings"

    def embed_text(self, text):
        # Custom embedding logic
        return [0.1, 0.2, 0.3]  # Example embedding vector

@hookimpl
def register_embedding_models(register):
    register(CustomEmbeddingModel(), aliases=["custom"])

# Using the plugins
from recall_kit.storage import SQLiteBackend
from recall_kit.embeddings import OpenAIEmbeddingService

# Create storage and embedding service instances
storage = SQLiteBackend()
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service
)
memory = recall.create_memory("This is an important fact to remember.")
results = recall.search("important information")
```

#### Approach 2: Component-based Plugin System

```python
from recall_kit import RecallKit
from recall_kit.components import EmbeddingModel, MemoryProcessor, MemoryStore

# Create custom components
class MyEmbeddingModel(EmbeddingModel):
    def embed(self, text):
        # Custom embedding logic
        return [0.1, 0.2, 0.3]  # Example embedding vector

class MyMemoryProcessor(MemoryProcessor):
    def process(self, memory_text):
        # Custom processing logic
        return f"Enhanced: {memory_text}"

class MyMemoryStore(MemoryStore):
    def __init__(self):
        self.memories = {}

    def store(self, memory_id, memory_data):
        self.memories[memory_id] = memory_data

    def retrieve(self, query_vector, limit=10):
        # Custom retrieval logic
        return list(self.memories.values())[:limit]

# Initialize RecallKit with custom components
# Create custom service instances
embedding_service = MyEmbeddingModel()
storage = MyMemoryStore()

# Initialize RecallKit with custom service instances
recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service
)

# Use the memory processor separately
processor = MyMemoryProcessor()

# Use the customized RecallKit
recall.create_memory("This is an important fact to remember.")
results = recall.search("important information")
```


#### Approach 3: Functional Component System

```python
from recall_kit import RecallKit
from typing import List, Dict, Any, Callable, Optional
from openai.types.chat import ChatCompletionRequest

# Define a memory source class
class MemorySource:
    def __init__(self, text: str, title: str, address: str):
        self.text = text
        self.title = title
        self.address = address

    def to_memory(self) -> Dict[str, Any]:
        """Convert source to memory format"""
        return {
            "text": self.text,
            "title": self.title,
            "address": self.address
        }

# Define core memory functions
def retrieve(query_text: str, request: ChatCompletionRequest) -> List[Dict[str, Any]]:
    """Retrieve memories based on query text and chat request context.

    This function is responsible for semantic search against the vector store.
    """
    # Custom retrieval logic
    return [
        {"text": "Memory about renewable energy", "score": 0.92},
        {"text": "Discussion about solar panels", "score": 0.85}
    ]

def filter_memories(memory: Dict[str, Any], request: ChatCompletionRequest) -> bool:
    """Filter memories for relevance to the current request.

    This optional function allows fine-grained control over which memories are included.
    """
    # Example filtering logic based on relevance threshold
    return memory.get("score", 0) > 0.8

def rerank_memories(memories: List[Dict[str, Any]], request: ChatCompletionRequest) -> List[Dict[str, Any]]:
    """Rerank memories based on additional criteria beyond vector similarity.

    This optional function allows custom sorting of retrieved memories.
    """
    # Example reranking logic
    return sorted(memories, key=lambda m: m.get("score", 0), reverse=True)

def augment_request(memories: List[Dict[str, Any]], request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Augment the chat completion request with retrieved memories.

    This function determines how memories are incorporated into the prompt.
    """
    # Create a copy of the request to avoid modifying the original
    augmented_request = dict(request)

    # Add memories to system message or create a new one
    memory_context = "Relevant memories:\n" + "\n".join([f"- {m['text']}" for m in memories])

    if "messages" in augmented_request:
        # Find system message if it exists
        system_msg_idx = next((i for i, msg in enumerate(augmented_request["messages"])
                              if msg.get("role") == "system"), None)

        if system_msg_idx is not None:
            # Append to existing system message
            augmented_request["messages"][system_msg_idx]["content"] += f"\n\n{memory_context}"
        else:
            # Insert new system message at the beginning
            augmented_request["messages"].insert(0, {
                "role": "system",
                "content": memory_context
            })

    return augmented_request

# Create storage and embedding service instances
from recall_kit.storage import SQLiteBackend, PostgresBackend
from recall_kit.embeddings import OpenAIEmbeddingService

def create_storage(storage_type: str = "sqlite", connection_string: Optional[str] = None) -> StorageBackend:
    """Create a storage backend instance.

    Supports:
    - sqlite (default): Uses sqlite3 with sqlite-vec for vector storage
    - postgres: Uses PostgreSQL with pgvector for vector storage
    """
    if storage_type == "postgres" and connection_string:
        # Setup postgres with pgvector
        return PostgresBackend(connection_string)
    else:
        # Default to sqlite with sqlite-vec
        return SQLiteBackend("memories.db")

# Create service instances
storage = create_storage("sqlite")
embedding_service = OpenAIEmbeddingService(model_name="text-embedding-3-small")

# Initialize RecallKit with functional components
recall = RecallKit(
    storage=storage,
    embedding_service=embedding_service,
    retrieve_fn=retrieve,
    filter_fn=filter_memories,  # Optional
    rerank_fn=rerank_memories,  # Optional
    augment_fn=augment_request
)

# Use the customized RecallKit
recall.create_memory("This is an important fact to remember.")
results = recall.search("important information")
```

**Pros and Cons:**

**Approach 1: Hook-based Plugin System**
- Pros:
  - Similar to existing systems like LLM, familiar to users
  - Clear extension points with well-defined interfaces
  - Plugins can be distributed as separate packages
- Cons:
  - More complex implementation for the core package
  - May require more boilerplate code for simple customizations
  - Potential for plugin conflicts

**Approach 2: Component-based Plugin System**
- Pros:
  - Object-oriented approach with clear inheritance
  - Strong typing and interface definitions
  - Flexible composition of components
- Cons:
  - Requires understanding of class hierarchy
  - May be overkill for simple customizations
  - More verbose for users

**Approach 3: Functional Component System**
- Pros:
  - Simpler to implement for basic customizations
  - No need to understand class hierarchies
  - More concise and focused on behavior
  - Easier to test individual functions
- Cons:
  - Less structured than class-based approaches
  - May be harder to maintain state between operations
  - Potentially less clear interface boundaries



### OpenAI-Compatible Model Wrapper

```python
# Server implementation
from recall_kit.server import create_app

app = create_app(
    embedding_model="text-embedding-3-small",
    memory_db_path="./memories.db",
    auto_consolidate=True
)

# Run with any ASGI server
# uvicorn recall_kit.server:app --host 0.0.0.0 --port 8000

# Client usage
import openai

# Point to the Recall Kit server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-but-required-by-client"
)

# Use like a regular OpenAI client
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with memory capabilities."},
        {"role": "user", "content": "What do you remember about our previous conversations?"}
    ]
)

print(response.choices[0].message.content)
```

