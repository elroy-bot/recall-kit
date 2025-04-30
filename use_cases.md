# recall-kit Use Cases

This document outlines various use cases for recall-kit, demonstrating its flexibility and power in different scenarios.

## 1. Conversational AI with Memory

### Problem

Standard LLMs have limited context windows and no persistent memory, making it difficult to maintain coherent, long-term conversations that reference past interactions.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.integrations import MemoryEnabledModel

# Initialize memory manager
memory = MemoryManager()

# Create a memory-enabled model
model = MemoryEnabledModel(base_model="gpt-4")

def chat(user_message):
    # Store the user's message as a memory
    memory.store(
        user_message,
        metadata={"role": "user", "timestamp": current_time_iso()}
    )

    # Generate a response with memory context
    response = model.generate(user_message)

    # Store the assistant's response as a memory
    memory.store(
        response,
        metadata={"role": "assistant", "timestamp": current_time_iso()}
    )

    return response
```

### Benefits

- Conversations can span multiple sessions
- The AI can recall specific details mentioned hours or days ago
- Users don't need to repeat information they've already shared

## 2. Coding Assistant with Project Knowledge

### Problem

When working on large codebases, developers need assistance that understands the entire project context, not just the current file.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import FileSystemSource

# Initialize memory manager with file system source
memory = MemoryManager(
    source="file_system",
    source_config={
        "root_dir": "./project",
        "file_patterns": ["*.py", "*.js", "*.ts", "*.md"],
        "exclude_patterns": ["node_modules/*", "__pycache__/*"]
    }
)

# Index the project files
memory.index_files()

def code_assistant(query):
    # Retrieve relevant code snippets and documentation
    results = memory.recall(query)

    # Format the context for the LLM
    context = "Project context:\n\n"
    for result in results:
        context += f"File: {result.metadata.get('file_path')}\n"
        context += f"```{result.metadata.get('language', '')}\n"
        context += f"{result.content}\n"
        context += "```\n\n"

    # Generate a response with the project context
    prompt = f"{context}\n\nQuery: {query}\n\nResponse:"
    return llm.generate(prompt)
```

### Benefits

- Code suggestions that align with the project's architecture and style
- Answers that reference project-specific implementations
- Reduced need to manually search through the codebase

## 3. Document Q&A System

### Problem

Organizations have large collections of documents (manuals, reports, policies) that employees need to query efficiently.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import DocumentSource
from recall_kit.processors import ChunkProcessor

# Initialize memory manager with document source
memory = MemoryManager(
    source="document",
    source_config={
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
)

# Process and index documents
documents = [
    {"path": "policies/hr_manual.pdf", "metadata": {"department": "HR"}},
    {"path": "reports/q1_2025.docx", "metadata": {"type": "quarterly_report"}},
    {"path": "manuals/product_specs.pdf", "metadata": {"product": "widget_x"}}
]

for doc in documents:
    memory.index_document(doc["path"], metadata=doc["metadata"])

def answer_question(query):
    # Retrieve relevant document chunks
    results = memory.recall(query)

    # Format the context for the LLM
    context = "Document context:\n\n"
    for result in results:
        context += f"Source: {result.metadata.get('source')}\n"
        context += f"Section: {result.metadata.get('section', 'N/A')}\n"
        context += f"{result.content}\n\n"

    # Generate a response with the document context
    prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.generate(prompt)
```

### Benefits

- Fast, accurate answers from organizational knowledge
- Reduced time spent searching through documents
- Consistent responses based on official documentation

## 4. Personal Knowledge Management

### Problem

Individuals struggle to organize and retrieve their personal notes, ideas, and learnings effectively.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.integrations import CLIApp

# Initialize memory manager
memory = MemoryManager(
    source="vector_store",
    source_config={
        "persist_directory": "~/.recall_kit/personal_knowledge"
    }
)

# Create CLI application
app = CLIApp(memory)

# Usage examples:

# Store a note
# $ recall-note "I learned that Python's GIL prevents true multi-threading"

# Store a bookmark
# $ recall-bookmark https://example.com/article "Great article about distributed systems"

# Query your knowledge
# $ recall-query "What did I learn about Python threading?"

# The CLI app handles these commands and uses the memory manager to store and retrieve information
```

### Benefits

- Unified interface for personal knowledge management
- Semantic search across all personal notes and bookmarks
- Command-line interface for quick capture and retrieval

## 5. Content Recommendation System

### Problem

Content platforms need to recommend relevant articles, products, or media based on user preferences and behavior.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.rankers import PersonalizationRanker

# Initialize memory manager for content items
content_memory = MemoryManager(
    source="vector_store",
    ranker=PersonalizationRanker()
)

# Initialize memory manager for user preferences
user_memory = MemoryManager(
    source="in_memory"
)

# Index content items
for item in content_items:
    content_memory.store(
        item["description"],
        metadata={
            "id": item["id"],
            "title": item["title"],
            "category": item["category"],
            "tags": item["tags"]
        }
    )

def track_user_interaction(user_id, item_id, interaction_type):
    # Store user interaction as a memory
    user_memory.store(
        f"User interacted with item {item_id}",
        metadata={
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type,
            "timestamp": current_time_iso()
        }
    )

def get_recommendations(user_id, count=5):
    # Retrieve user preferences
    user_prefs = user_memory.recall(
        f"What does user {user_id} like?",
        filter_config={"user_id": user_id}
    )

    # Generate a query based on user preferences
    if user_prefs:
        # Extract preferences from user interactions
        categories = []
        tags = []
        for pref in user_prefs:
            item_id = pref.metadata.get("item_id")
            if item_id:
                item = get_item_by_id(item_id)
                if item:
                    categories.append(item["category"])
                    tags.extend(item["tags"])

        # Create a preference-based query
        query = " ".join(categories + tags)
    else:
        # Default query for new users
        query = "popular trending"

    # Get recommendations based on the query
    recommendations = content_memory.recall(
        query,
        limit=count
    )

    return [
        {
            "id": rec.metadata.get("id"),
            "title": rec.metadata.get("title"),
            "relevance": rec.relevance
        }
        for rec in recommendations
    ]
```

### Benefits

- Personalized recommendations based on user behavior
- Content discovery that improves over time
- Flexible integration with existing content platforms

## 6. Automated Documentation Generation

### Problem

Keeping documentation in sync with code changes is time-consuming and often neglected.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import CodebaseSource
from recall_kit.reflectors import DocumentationReflector

# Initialize memory manager for codebase
code_memory = MemoryManager(
    source="codebase",
    source_config={
        "root_dir": "./project",
        "file_patterns": ["*.py", "*.js", "*.ts"],
        "exclude_patterns": ["node_modules/*", "__pycache__/*"]
    },
    reflector=DocumentationReflector()
)

# Index the codebase
code_memory.index_codebase()

def generate_documentation():
    # Generate documentation for each module
    modules = code_memory.get_modules()
    documentation = {}

    for module in modules:
        # Retrieve code context for the module
        module_code = code_memory.recall(
            f"code in module {module}",
            filter_config={"module": module}
        )

        # Generate documentation using reflection
        doc = code_memory.reflect(
            module_code,
            query=f"Generate documentation for module {module}"
        )

        documentation[module] = doc

    # Write documentation to files
    for module, doc in documentation.items():
        with open(f"docs/{module}.md", "w") as f:
            f.write(doc)
```

### Benefits

- Documentation that stays in sync with code
- Consistent documentation style across the project
- Reduced manual effort for developers

## 7. Continuous Learning System

### Problem

AI systems need to learn from new data and experiences over time to remain relevant and accurate.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.filters import FeedbackFilter

# Initialize memory manager with feedback-aware configuration
memory = MemoryManager(
    source="vector_store",
    filters=[FeedbackFilter(min_rating=3)]
)

def process_user_interaction(query, response, feedback=None):
    # Store the interaction
    memory.store(
        f"Q: {query}\nA: {response}",
        metadata={
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": current_time_iso()
        }
    )

    # If negative feedback was provided, store it for learning
    if feedback and feedback < 3:
        memory.store(
            f"Improvement needed for query: {query}",
            metadata={
                "query": query,
                "response": response,
                "feedback": feedback,
                "type": "improvement_opportunity"
            }
        )

def generate_improved_response(query):
    # Find similar past queries
    similar_interactions = memory.recall(query)

    # Find improvement opportunities
    improvement_opportunities = memory.recall(
        query,
        filter_config={"type": "improvement_opportunity"}
    )

    # Generate context for the LLM
    context = "Past interactions:\n\n"
    for interaction in similar_interactions:
        context += f"{interaction.content}\n"
        context += f"Feedback: {interaction.metadata.get('feedback', 'None')}\n\n"

    if improvement_opportunities:
        context += "Areas for improvement:\n\n"
        for opportunity in improvement_opportunities:
            context += f"{opportunity.content}\n\n"

    # Generate improved response
    prompt = f"{context}\n\nNew query: {query}\n\nImproved response:"
    return llm.generate(prompt)
```

### Benefits

- AI system that learns from user feedback
- Continuous improvement over time
- Adaptation to changing user needs and expectations

## 8. Research Assistant

### Problem

Researchers need to organize and retrieve information from papers, notes, and experiments.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import ResearchSource

# Initialize memory manager for research materials
research_memory = MemoryManager(
    source="research",
    source_config={
        "papers_dir": "./papers",
        "notes_dir": "./notes",
        "experiments_dir": "./experiments"
    }
)

# Index research materials
research_memory.index_papers()
research_memory.index_notes()
research_memory.index_experiments()

def research_assistant(query):
    # Retrieve relevant research materials
    results = research_memory.recall(query)

    # Categorize results
    papers = []
    notes = []
    experiments = []

    for result in results:
        material_type = result.metadata.get("type")
        if material_type == "paper":
            papers.append(result)
        elif material_type == "note":
            notes.append(result)
        elif material_type == "experiment":
            experiments.append(result)

    # Format the response
    response = f"Research findings for: {query}\n\n"

    if papers:
        response += "Relevant papers:\n"
        for paper in papers:
            response += f"- {paper.metadata.get('title')} ({paper.metadata.get('authors')})\n"
            response += f"  Key finding: {paper.content[:200]}...\n\n"

    if notes:
        response += "Your notes:\n"
        for note in notes:
            response += f"- {note.metadata.get('title')} ({note.metadata.get('date')})\n"
            response += f"  {note.content[:200]}...\n\n"

    if experiments:
        response += "Related experiments:\n"
        for exp in experiments:
            response += f"- {exp.metadata.get('title')} ({exp.metadata.get('date')})\n"
            response += f"  Results: {exp.metadata.get('results', 'N/A')}\n\n"

    return response
```

### Benefits

- Unified access to research materials
- Quick retrieval of relevant information
- Connections between related papers, notes, and experiments

## 9. Directory Watcher for Automatic Ingestion

### Problem

Teams need to keep their knowledge base up-to-date as new documents are added to shared directories.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import DirectoryWatcherSource
import time

# Initialize memory manager with directory watcher
memory = MemoryManager(
    source="directory_watcher",
    source_config={
        "watch_dirs": ["./shared_docs", "./project_reports"],
        "file_patterns": ["*.pdf", "*.docx", "*.md", "*.txt"],
        "polling_interval": 300  # seconds
    }
)

# Start the watcher
memory.start_watching()

# In a real application, you would keep this running
try:
    while True:
        # Check for newly processed files
        new_files = memory.get_recently_processed_files()
        if new_files:
            print(f"Newly processed files: {new_files}")

        # Sleep to avoid high CPU usage
        time.sleep(60)
except KeyboardInterrupt:
    # Stop the watcher when the application exits
    memory.stop_watching()
```

### Benefits

- Automatic ingestion of new documents
- Always up-to-date knowledge base
- No manual indexing required

## 10. Multi-Agent Collaboration with Shared Memory

### Problem

Multiple AI agents need to collaborate and share knowledge to solve complex problems.

### Solution with recall-kit

```python
from recall_kit import MemoryManager
from recall_kit.sources import SharedMemorySource

# Initialize shared memory
shared_memory = MemoryManager(
    source="shared_memory",
    source_config={
        "namespace": "team_collaboration"
    }
)

class Agent:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
        self.memory = shared_memory  # All agents use the same shared memory

    def process_task(self, task):
        # Retrieve relevant information from shared memory
        context = self.memory.recall(
            f"{task} {self.specialty}",
            filter_config={"relevant_to": self.specialty}
        )

        # Generate a response based on the agent's specialty
        response = f"Agent {self.name} ({self.specialty}) response:\n"
        response += llm.generate(
            f"Task: {task}\nSpecialty: {self.specialty}\nContext: {[c.content for c in context]}\nResponse:"
        )

        # Store the agent's contribution to shared memory
        self.memory.store(
            response,
            metadata={
                "agent": self.name,
                "specialty": self.specialty,
                "task": task,
                "timestamp": current_time_iso()
            }
        )

        return response

# Create a team of specialized agents
team = [
    Agent("Alice", "data_analysis"),
    Agent("Bob", "software_engineering"),
    Agent("Charlie", "product_management"),
    Agent("Diana", "user_experience")
]

def solve_problem(problem):
    # Each agent contributes to the solution
    solutions = []
    for agent in team:
        solution = agent.process_task(problem)
        solutions.append(solution)

    # Generate a final integrated solution
    integrated_solution = shared_memory.reflect(
        shared_memory.recall(problem),
        query=f"Integrate all agent solutions for: {problem}"
    )

    return integrated_solution
```

### Benefits

- Knowledge sharing between specialized agents
- Collaborative problem-solving
- Integrated solutions that leverage multiple perspectives
