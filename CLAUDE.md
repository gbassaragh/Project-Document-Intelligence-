# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A GraphRAG (Graph + Retrieval-Augmented Generation) knowledge system that ingests project documents (3,500+ PDFs like PAFs, SRFs, IFRs) and structured data into a Neo4j knowledge graph, enabling complex multi-hop natural language queries through vector similarity search and graph traversal.

**Tech Stack**: Python 3.11+, Neo4j 5.x, OpenAI GPT-4o, LangChain, sentence-transformers

## Common Development Commands

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (required before running)
cp .env.example .env
# Edit .env with Neo4j credentials and OpenAI API key

# Verify installation
python verify_installation.py
```

### Running the Application
```bash
# Start the main application (interactive menu)
python main.py

# Run specific pipeline stages programmatically:
# 1. Initialize database schema
# 2. Ingest PDFs
# 3. Extract entities
# 4. Generate embeddings
# 5. Query the knowledge graph
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit              # Fast unit tests only
pytest -m integration       # Integration tests (requires Neo4j)
pytest -m asyncio           # Async tests

# Run tests in a specific module
pytest tests/test_ingestion/
pytest tests/test_extraction/

# Run single test file
pytest test_phase3_pipeline.py

# Coverage report (target: 70%+)
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view detailed coverage
```

### Code Quality
```bash
# Linting (PEP 8 compliance, max line length: 100)
flake8

# Type checking
mypy src/

# Format code (if black is used)
black src/ tests/
```

### Neo4j Management
```bash
# Verify Neo4j connection
python -c "from src.database.connection import get_connection; get_connection().verify_connection()"

# Start Neo4j via Docker
docker run --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest

# Access Neo4j Browser: http://localhost:7474
```

## Architecture

### High-Level Pipeline Flow
```
Data Sources → Ingestion → Entity Extraction → Vector Indexing → Query Interface
     ↓              ↓               ↓                   ↓               ↓
  PDFs/CSV    Parse/Chunk    LLM-based NER      Embeddings    Hybrid Retrieval
              → Neo4j        → Relationships     → Vector      (Vector + Graph)
                             → Graph              Index        → LLM Answer
```

### Core Components

**1. Configuration Layer** (`src/config/settings.py`)
- Pydantic-based settings with validation
- All configuration loaded from `.env` file
- Config categories: Neo4j, OpenAI, Data paths, Processing, Logging, Extraction, Quality
- Settings are type-safe and validated on load

**2. Database Layer** (`src/database/`)
- `connection.py`: Neo4j connection pooling (singleton pattern)
- `schema.py`: Graph schema initialization with constraints and vector indexes
- Node types: Project, Document, Chunk, Person, Team, Procedure, Deliverable
- Relationships: MANAGES, BELONGS_TO, HAS_DOCUMENT, HAS_CHUNK, MENTIONS_*, REFERENCES, GENERATED_FOR

**3. Ingestion Layer** (`src/ingestion/`)
- `pdf_parser.py`: PDF parsing with PyPDF, text chunking (configurable size/overlap)
- Creates Document and Chunk nodes in Neo4j
- Metadata preservation: file path, page count, document type

**4. Extraction Layer** (`src/extraction/`)
- `entity_extractor.py`: Synchronous LLM-based entity extraction
- `entity_extractor_async.py`: Async version with worker pool for parallel processing
- `paf_extractor.py`: Specialized extractor for PAF (Project Authorization Form) documents
- `prompt_manager.py`: Versioned prompts with fallback chain (GPT-4o → GPT-4o-mini)
- Extracts entities (Person, Project, Procedure, etc.) and relationships from chunks
- Retry logic with exponential backoff for API failures

**5. RAG Layer** (`src/rag/`)
- `embeddings.py`: Vector embedding generation using sentence-transformers
- `query_interface.py`: Hybrid query system combining:
  - Vector similarity search (cosine similarity on embeddings)
  - Graph traversal for context expansion
  - LLM synthesis for final answer generation
- Returns answers with confidence scores, source citations, and related entities

**6. Models Layer** (`src/models/`)
- `paf_document.py`: Pydantic models for structured PAF data
- `validation.py`: Field validators and quality checks
- Confidence scoring based on field completeness

**7. Validation Layer** (`src/validation/`)
- `document_validator.py`: Document quality validation
- Confidence and completeness thresholds
- Failed documents stored in `failed_loads_dir` for review

**8. Testing Layer** (`src/testing/`)
- `golden_test_generator.py`: Auto-generates regression tests from successful extractions
- Creates ground truth datasets for quality monitoring

### Key Architectural Patterns

**Configuration-Driven Design**
- All settings externalized to `.env` file
- Pydantic validation ensures type safety and required fields
- Environment-specific configurations (dev/prod) via different `.env` files

**Batch Processing Strategy**
- Configurable batch sizes via `BATCH_SIZE` env var
- Memory-efficient processing of large document collections
- Neo4j MERGE with UNWIND for bulk operations

**Error Handling & Resilience**
- Three-tier retry logic with exponential backoff
- Model degradation: GPT-4o → GPT-4o-mini on failures
- Failed documents isolated to dedicated directories
- Comprehensive logging at all pipeline stages

**Quality Assurance**
- Confidence scoring on extracted data
- Completeness thresholds for validation
- Golden test auto-generation for regression prevention
- 70%+ test coverage requirement

**Async Processing**
- Worker pool pattern for parallel LLM calls (`entity_extractor_async.py`)
- Controlled concurrency to avoid rate limits
- Thread-safe Neo4j operations

## Important Implementation Details

### Neo4j Schema Management
- Schema initialization creates constraints (uniqueness on Project.id, Person.name, etc.)
- Vector index created automatically with configurable dimension (default: 384 for all-MiniLM-L6-v2)
- Schema changes should go through `SchemaManager.initialize_schema()`

### Entity Extraction Flow
1. Retrieve chunks from Neo4j (batched)
2. Send to LLM with structured extraction prompt
3. Parse JSON response into entities and relationships
4. Validate against Pydantic models
5. Merge into Neo4j graph (MERGE operations prevent duplicates)
6. Log extraction metrics (success rate, entity counts)

### Embedding Pipeline
- Sentence-transformers model loaded once and reused
- Batch embedding generation for efficiency
- Embeddings stored as properties on Chunk nodes
- Vector index enables fast similarity search

### Query Interface Pattern
```python
# Hybrid query combines three steps:
# 1. Vector search: Find semantically similar chunks
# 2. Graph expansion: Traverse relationships for context
# 3. LLM synthesis: Generate coherent answer from context
result = query_interface.hybrid_query("question", top_k=5)
# Returns: {answer, confidence, chunks, context}
```

### Logging Strategy
- All modules use Python's standard logging
- Log levels: DEBUG (development), INFO (production), WARNING/ERROR (issues)
- Logs written to both file (`logs/graphrag.log`) and stdout
- Exception tracebacks preserved with `exc_info=True`

## Development Workflow

### Adding a New Document Type
1. Create Pydantic model in `src/models/`
2. Add specialized extractor in `src/extraction/` (follow `paf_extractor.py` pattern)
3. Update schema in `src/database/schema.py` if new node types needed
4. Add extraction prompts in `prompts/` directory
5. Register in `prompt_manager.py` for version control
6. Write unit tests in `tests/test_extraction/`

### Modifying Extraction Logic
- All extraction prompts versioned via `EXTRACTION_PROMPT_VERSION` env var
- Prompts stored in `prompts/` directory
- Test changes with `AUTO_GENERATE_GOLDEN_TESTS=true` to create regression tests
- Monitor extraction quality via confidence scores

### Optimizing Performance
- Adjust `BATCH_SIZE` based on available memory
- Use async extractors for large document sets
- Monitor Neo4j query performance via EXPLAIN/PROFILE
- Consider GPU acceleration for embeddings (sentence-transformers[gpu])

## Testing Philosophy

- **Unit tests** (`-m unit`): Fast, no external dependencies, mock Neo4j/OpenAI
- **Integration tests** (`-m integration`): Require running Neo4j, test end-to-end flows
- **Async tests** (`-m asyncio`): Test concurrent processing and race conditions
- Test coverage must be ≥70% (enforced in pytest.ini)
- Golden tests auto-generated from production extractions for regression detection

## Common Pitfalls

1. **Missing .env file**: Application will fail validation on startup
2. **Neo4j not running**: Connection errors on any database operation
3. **OpenAI rate limits**: Use async extractors with proper backoff, enable model degradation
4. **Memory issues with large PDFs**: Reduce `BATCH_SIZE`, process in smaller batches
5. **Vector dimension mismatch**: Ensure `VECTOR_DIMENSION` matches embedding model output
6. **Duplicate entities**: MERGE operations handle this, but check uniqueness constraints

## Configuration Reference

Critical `.env` settings:
- `NEO4J_PASSWORD`: Must be set, no default
- `OPENAI_API_KEY`: Required for entity extraction and query answering
- `OPENAI_MODEL`: Default `gpt-4o`, fallback to `gpt-4o-mini` on degradation
- `BATCH_SIZE`: Default 100, tune based on memory
- `CONFIDENCE_THRESHOLD`: Minimum confidence score (0-10) for accepting extractions
- `COMPLETENESS_THRESHOLD`: Fraction of required fields that must be present (0-1)

See `.env.example` for complete configuration options.
