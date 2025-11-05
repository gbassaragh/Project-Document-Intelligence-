# GraphRAG Knowledge System

A sophisticated knowledge graph and Retrieval-Augmented Generation (RAG) system for managing and querying large collections of project documents and structured data.

## 🎯 Overview

The GraphRAG Knowledge System ingests 3,500+ unstructured project documents (PDFs like PAFs, SRFs, IFRs) and structured data (Excel/CSV files) into a unified Neo4j knowledge graph. It enables complex, multi-hop natural language queries by combining:

- **Vector Similarity Search**: Semantic search using embeddings
- **Knowledge Graph Traversal**: Relationship-based context expansion
- **LLM-Powered Answers**: GPT-4o generates comprehensive answers from retrieved context

### Example Query
> "Find all projects that reference 'AACE procedures', had an 'IFR' issued, and were managed by someone from David Smith's team. Summarize the key risks mentioned in those IFRs."

## 🏗️ Architecture

### Technology Stack
- **Language**: Python 3.11+
- **Graph Database**: Neo4j 5.x
- **GraphRAG Library**: `neo4j-graphrag-python`
- **Data Preprocessing**: DuckDB (in-memory)
- **Document Parsing**: PyPDF
- **LLM Orchestration**: LangChain + OpenAI GPT-4o
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Configuration**: python-dotenv

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
├─────────────────────────────────────────────────────────────┤
│  • Structured Data (Excel/CSV) → DuckDB → Neo4j             │
│  • PDF Documents → PyPDF → Text Chunks → Neo4j              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Entity Extraction Layer                    │
├─────────────────────────────────────────────────────────────┤
│  • LLM-based entity extraction (GPT-4o)                      │
│  • Extract: Person, Project, Procedure, Deliverable         │
│  • Create relationships: MANAGES, MENTIONS, REFERENCES       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Vector Indexing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  • Generate embeddings for all text chunks                   │
│  • Create Neo4j vector index for similarity search           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Query Interface                         │
├─────────────────────────────────────────────────────────────┤
│  • Hybrid query: Vector search + Graph traversal             │
│  • Context expansion using relationships                     │
│  • LLM-generated answers with citations                      │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Schema

### Nodes
- **Project** `{id, name, status}`
- **Document** `{id, type, path, file_name, num_pages}`
- **Chunk** `{chunk_id, text, embedding, chunk_index}`
- **Person** `{name, role}`
- **Team** `{name}`
- **Procedure** `{id}`
- **Deliverable** `{name, file_type}`

### Relationships
- `(Person)-[:MANAGES]->(Project)`
- `(Person)-[:BELONGS_TO]->(Team)`
- `(Project)-[:HAS_DOCUMENT]->(Document)`
- `(Document)-[:HAS_CHUNK]->(Chunk)`
- `(Chunk)-[:MENTIONS_PERSON]->(Person)`
- `(Chunk)-[:MENTIONS_PROCEDURE]->(Procedure)`
- `(Document)-[:GENERATED_FOR]->(Project)`

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Neo4j 5.x (Community or Enterprise)
- OpenAI API key

### 1. Clone Repository
```bash
cd /root/myprojects/Project_Document_Intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o

# Data Directories
STRUCTURED_DATA_DIR=./data/structured
PDF_DATA_DIR=./data/pdfs
OUTPUT_DIR=./output
```

### 4. Prepare Data
Place your data files in the appropriate directories:

```
data/
├── structured/
│   ├── projects.xlsx
│   ├── teams.xlsx
│   └── managers.csv
└── pdfs/
    ├── project_paf_001.pdf
    ├── project_srf_002.pdf
    └── project_ifr_003.pdf
```

## 📖 Usage

### Run the Application
```bash
python main.py
```

### Menu Options

**1. Run Full Ingestion Pipeline**
- Initializes database schema
- Ingests structured data (Excel/CSV)
- Parses and ingests PDF documents
- Extracts entities and relationships
- Generates embeddings and vector index

**2-6. Individual Pipeline Steps**
- Run specific steps separately for incremental updates

**7. Run Example Queries**
- Executes predefined example queries to demonstrate capabilities

**8. Interactive Query Mode**
- Enter custom natural language questions
- Get answers with source citations

### Example Queries

```python
from src.rag.query_interface import GraphRAGQuery
from src.database.connection import get_connection

connection = get_connection()
query_interface = GraphRAGQuery(connection)

# Ask a question
answer = query_interface.answer_question(
    "What are the most common risks mentioned in IFR documents?"
)
print(answer)

# Get project summary
summary = query_interface.get_project_summary("Project Alpha")
print(summary)

# Hybrid query with full context
result = query_interface.hybrid_query(
    "Find projects managed by David Smith's team that mention AACE procedures"
)
print(result['answer'])
print(f"Confidence: {result['confidence']}")
```

## 🔧 Configuration

### Processing Configuration
Adjust batch sizes and embedding models in `.env`:

```env
BATCH_SIZE=100
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
```

### Logging Configuration
```env
LOG_LEVEL=INFO
LOG_FILE=./logs/graphrag.log
```

## 🏛️ Project Structure

```
Project_Document_Intelligence/
├── src/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── database/
│   │   ├── connection.py        # Neo4j connection pooling
│   │   └── schema.py            # Schema initialization
│   ├── ingestion/
│   │   ├── structured_data.py   # Excel/CSV ingestion with DuckDB
│   │   └── pdf_parser.py        # PDF parsing and chunking
│   ├── extraction/
│   │   └── entity_extractor.py  # LLM-based entity extraction
│   ├── rag/
│   │   ├── embeddings.py        # Vector embeddings management
│   │   └── query_interface.py   # Hybrid query interface
│   └── utils/
├── data/
│   ├── structured/              # Excel/CSV files
│   └── pdfs/                    # PDF documents
├── logs/                        # Application logs
├── output/                      # Output files
├── tests/                       # Unit tests
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🎓 Design Principles

### Code Quality
- **Modular**: Clear separation of concerns
- **Strongly Typed**: Type hints throughout
- **PEP 8 Compliant**: Consistent code style

### Maintainability
- **Configuration-Driven**: All settings in `.env`
- **Logging**: Comprehensive logging at all levels
- **Error Handling**: Graceful error recovery

### Data Integrity
- **Constraints**: Uniqueness constraints on all primary entities
- **Validation**: Input validation at all stages
- **Batch Processing**: Optimized MERGE and UNWIND operations

### Efficiency
- **In-Memory Processing**: DuckDB for structured data preprocessing
- **Batch Operations**: Configurable batch sizes for all operations
- **Vector Index**: Fast similarity search with cosine similarity

### Testability
- **Main Block**: Example queries validate end-to-end functionality
- **Modular Design**: Each component independently testable
- **Logging**: Detailed execution traces for debugging

## 🔍 Advanced Features

### Hybrid Retrieval
Combines vector similarity search with graph-based context expansion:
1. Vector search finds semantically similar chunks
2. Graph traversal expands context with related entities
3. LLM synthesizes comprehensive answer

### Multi-Hop Queries
The knowledge graph enables complex queries spanning multiple relationships:
- "Projects → Managers → Teams"
- "Documents → Procedures → Standards"
- "Projects → Documents → Risks"

### Entity Extraction
LLM-powered extraction identifies:
- **Entities**: Person, Project, Procedure, Deliverable, Risk
- **Relationships**: MANAGES, MENTIONS, REFERENCES, HAS_RISK
- **Context**: Surrounding text for each entity

## 🐛 Troubleshooting

### Neo4j Connection Issues
```bash
# Verify Neo4j is running
neo4j status

# Check connection
python -c "from src.database.connection import get_connection; get_connection().verify_connection()"
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### OpenAI API Errors
- Verify API key is correct in `.env`
- Check API quota and billing status
- Ensure model name is correct (e.g., `gpt-4o`)

### Memory Issues
- Reduce `BATCH_SIZE` in `.env`
- Process documents in smaller batches
- Use GPU for embeddings if available

## 📈 Performance Optimization

### Embedding Generation
- Use GPU acceleration: `pip install sentence-transformers[gpu]`
- Increase batch size for faster processing

### Neo4j Optimization
- Allocate sufficient memory (4GB+ recommended)
- Create indexes on frequently queried properties
- Use APOC procedures for complex operations

### Query Performance
- Adjust `top_k` parameter for fewer/more results
- Tune `score_threshold` to filter low-quality matches
- Use Cypher PROFILE to analyze slow queries

## 📝 License

This project is proprietary and confidential.

## 🤝 Contributing

Internal project - please follow the established code quality standards and principles outlined in `CLAUDE.md`.

## 📧 Support

For issues or questions, please contact the development team.

---

**Built with ❤️ using Python, Neo4j, LangChain, and OpenAI GPT-4o**
