# How recall-kit Works

This document explains the core concepts of how recall-kit stores, processes, and retrieves memories.

## Memory Storage

### Memory Structure

In recall-kit, a memory consists of:

```
┌─────────────────────────────────┐
│ Memory                          │
├─────────────────────────────────┤
│ • Content (text)                │
│ • Metadata (key-value pairs)    │
│ • Embedding (vector)            │
│ • ID (unique identifier)        │
│ • Timestamp                     │
└─────────────────────────────────┘
```

- **Content**: The actual text of the memory
- **Metadata**: Additional information about the memory (e.g., source, category, tags)
- **Embedding**: Vector representation of the memory for semantic search
- **ID**: Unique identifier for the memory
- **Timestamp**: When the memory was created or last updated

### Storage Backends

recall-kit supports multiple storage backends through its plugin system:

1. **In-Memory Storage**: Simple storage in RAM (default)
2. **Vector Stores**: For semantic search capabilities
   - Supports various vector databases (FAISS, Chroma, Pinecone, etc.)
3. **File-Based Storage**: Persistent storage in files
4. **Database Storage**: SQL or NoSQL database storage
5. **Custom Storage**: Create your own storage backend

When a memory is stored:

```python
# Simplified internal process
def store_memory(content, metadata=None):
    # 1. Generate a unique ID
    memory_id = generate_uuid()

    # 2. Create timestamp
    timestamp = current_time_iso()

    # 3. Generate embedding if using vector store
    embedding = None
    if using_vector_store():
        embedding = embedding_model.embed(content)

    # 4. Create memory object
    memory = {
        "id": memory_id,
        "content": content,
        "metadata": metadata or {},
        "embedding": embedding,
        "timestamp": timestamp
    }

    # 5. Store in the selected backend
    storage_backend.store(memory)

    return memory_id
```

## Memory Retrieval

### Retrieval Process

When retrieving memories, recall-kit follows this process:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │     │             │
│  Generate   │────▶│  Retrieve   │────▶│   Filter    │────▶│   Rerank    │────▶│   Reflect   │
│  Query      │     │  Candidates │     │  Memories   │     │  Results    │     │  (Optional) │
│             │     │             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Generate Query**: Process the input query
2. **Retrieve Candidates**: Get potential matches from storage
3. **Filter Memories**: Remove irrelevant or low-quality memories
4. **Rerank Results**: Order by relevance or importance
5. **Reflect**: Optionally generate meta-information about results

### Retrieval Methods

recall-kit supports multiple retrieval methods:

#### 1. Semantic Search (Default)

Uses embeddings to find semantically similar memories:

```python
def semantic_search(query, memories, top_k=10):
    # 1. Generate query embedding
    query_embedding = embedding_model.embed(query)

    # 2. Calculate similarity scores
    scores = []
    for memory in memories:
        if memory["embedding"] is not None:
            similarity = cosine_similarity(query_embedding, memory["embedding"])
            scores.append((memory, similarity))

    # 3. Sort by similarity and return top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [memory for memory, _ in scores[:top_k]]
```

#### 2. Keyword Search

Uses text-based search for finding exact matches:

```python
def keyword_search(query, memories, top_k=10):
    # Simple keyword matching
    keywords = extract_keywords(query)
    scores = []

    for memory in memories:
        score = 0
        for keyword in keywords:
            if keyword.lower() in memory["content"].lower():
                score += 1

        if score > 0:
            scores.append((memory, score))

    # Sort by score and return top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [memory for memory, _ in scores[:top_k]]
```

#### 3. Hybrid Search

Combines semantic and keyword search for better results:

```python
def hybrid_search(query, memories, top_k=10):
    semantic_results = semantic_search(query, memories, top_k=top_k*2)
    keyword_results = keyword_search(query, memories, top_k=top_k*2)

    # Combine and deduplicate results
    combined = {}
    for memory in semantic_results:
        combined[memory["id"]] = {"memory": memory, "score": 0.7}

    for memory in keyword_results:
        if memory["id"] in combined:
            combined[memory["id"]]["score"] += 0.3
        else:
            combined[memory["id"]] = {"memory": memory, "score": 0.3}

    # Sort by combined score
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [item["memory"] for item in sorted_results[:top_k]]
```

## Memory Filtering

After retrieving candidate memories, recall-kit applies filters to remove irrelevant or low-quality memories:

### Common Filters

#### 1. Relevance Threshold Filter

Removes memories below a certain relevance score:

```python
def relevance_filter(memories, threshold=0.7):
    return [
        memory for memory in memories
        if memory.get("relevance", 0) >= threshold
    ]
```

#### 2. Time Decay Filter

Reduces the importance of older memories:

```python
def time_decay_filter(memories, half_life_days=7):
    now = time.time()
    filtered = []

    for memory in memories:
        timestamp = parse_timestamp(memory.get("timestamp"))
        age_days = (now - timestamp) / (24 * 60 * 60)
        decay_factor = 2 ** (-age_days / half_life_days)

        # Apply decay to relevance score
        memory["relevance"] = memory.get("relevance", 1.0) * decay_factor
        filtered.append(memory)

    return filtered
```

#### 3. Duplicate Filter

Removes duplicate or nearly identical memories:

```python
def deduplication_filter(memories, similarity_threshold=0.95):
    unique_memories = []

    for memory in memories:
        is_duplicate = False
        for unique in unique_memories:
            similarity = text_similarity(memory["content"], unique["content"])
            if similarity > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_memories.append(memory)

    return unique_memories
```

## Memory Reranking

After filtering, recall-kit reranks the memories to present the most relevant ones first:

### Reranking Methods

#### 1. Relevance Ranker (Default)

Sorts memories by their relevance score:

```python
def relevance_ranker(memories):
    return sorted(
        memories,
        key=lambda x: x.get("relevance", 0),
        reverse=True
    )
```

#### 2. Recency Ranker

Prioritizes recent memories:

```python
def recency_ranker(memories):
    return sorted(
        memories,
        key=lambda x: parse_timestamp(x.get("timestamp", 0)),
        reverse=True
    )
```

#### 3. Hybrid Ranker

Combines multiple ranking factors:

```python
def hybrid_ranker(memories, relevance_weight=0.7, recency_weight=0.3):
    now = time.time()

    for memory in memories:
        # Normalize recency (0-1 scale, 1 being most recent)
        timestamp = parse_timestamp(memory.get("timestamp", 0))
        age = now - timestamp
        max_age = 30 * 24 * 60 * 60  # 30 days in seconds
        recency = 1 - min(age / max_age, 1)

        # Combine scores
        relevance = memory.get("relevance", 0)
        memory["combined_score"] = (
            relevance * relevance_weight +
            recency * recency_weight
        )

    return sorted(
        memories,
        key=lambda x: x.get("combined_score", 0),
        reverse=True
    )
```

## Reflection

The final optional step in the recall pipeline is reflection, which generates meta-information about the retrieved memories:

### Reflection Types

#### 1. Summary Reflection

Generates a summary of the retrieved memories:

```python
def summarize_memories(memories, query):
    if not memories:
        return None

    # Combine memory contents
    combined = "\n".join(memory["content"] for memory in memories)

    # Generate summary using an LLM
    prompt = f"""
    Based on these memories:
    {combined}

    Provide a concise summary relevant to: {query}
    """

    return llm.generate(prompt)
```

#### 2. Key Points Reflection

Extracts key points from the memories:

```python
def extract_key_points(memories, query):
    if not memories:
        return None

    # Combine memory contents
    combined = "\n".join(memory["content"] for memory in memories)

    # Extract key points using an LLM
    prompt = f"""
    Based on these memories:
    {combined}

    Extract 3-5 key points relevant to: {query}
    """

    return llm.generate(prompt)
```

## Memory Pipeline in Action

Here's how the entire pipeline works together:

```python
def recall(query, config=None):
    config = config or default_config()

    # 1. Retrieve candidate memories
    candidates = retrieve_memories(
        query,
        source=config["source"],
        method=config["retrieval_method"],
        limit=config["max_candidates"]
    )

    # 2. Apply filters
    filtered = candidates
    for filter_fn in config["filters"]:
        filtered = filter_fn(filtered)

    # 3. Rerank results
    ranked = config["ranker"](filtered)

    # 4. Apply limit
    limited = ranked[:config["max_results"]]

    # 5. Generate reflection (if enabled)
    reflection = None
    if config["generate_reflection"]:
        reflection = config["reflector"](limited, query)

        # 6. Store reflection as a new memory (if enabled)
        if config["store_reflections"] and reflection:
            store_memory(
                reflection,
                metadata={
                    "type": "reflection",
                    "query": query
                }
            )

    return {
        "memories": limited,
        "reflection": reflection
    }
```

## Customization

recall-kit's memory system is designed to be highly customizable:

1. **Storage**: Choose or create custom storage backends
2. **Retrieval**: Select or implement retrieval methods
3. **Filtering**: Configure or create custom filters
4. **Ranking**: Select or implement ranking algorithms
5. **Reflection**: Enable/disable or customize reflection generation

This flexibility allows you to tailor the memory system to your specific needs, whether you're building a chatbot, a knowledge management system, or any other application that benefits from memory capabilities.
