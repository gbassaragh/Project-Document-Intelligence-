# GraphRAG Knowledge System - Quick Start Guide

Get up and running with the GraphRAG Knowledge System in under 10 minutes.

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Neo4j 5.x installed and running
- [ ] OpenAI API key
- [ ] Project documents (PDFs) and structured data (Excel/CSV)

## Step-by-Step Setup

### 1. Install Neo4j

**Option A: Docker (Recommended)**
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

**Option B: Local Installation**
- Download from https://neo4j.com/download/
- Install and start Neo4j
- Set password via web interface at http://localhost:7474

### 2. Install Python Dependencies

```bash
cd /root/myprojects/Project_Document_Intelligence
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env
```

**Minimum Configuration:**
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o
```

### 4. Prepare Your Data

Create sample data structure:

```bash
# Create sample project data
mkdir -p data/structured
cat > data/structured/projects.xlsx
# Or copy your actual Excel/CSV files

# Add PDF documents
mkdir -p data/pdfs
# Copy your PDF documents (PAFs, SRFs, IFRs) to this directory
```

**Sample Structured Data Format:**

**projects.xlsx** or **projects.csv**:
| id | name | status | manager |
|----|------|--------|---------|
| PROJ-001 | Alpha Project | Active | John Doe |
| PROJ-002 | Beta Project | Completed | Jane Smith |

**teams.xlsx** or **teams.csv**:
| team_name | member_name |
|-----------|-------------|
| Engineering | John Doe |
| Engineering | Alice Johnson |
| Design | Jane Smith |

### 5. Run the Application

```bash
python main.py
```

**Select Option 1**: Run full ingestion pipeline

This will:
1. ✅ Initialize Neo4j schema with constraints and indexes
2. ✅ Ingest structured data from Excel/CSV files
3. ✅ Parse and chunk PDF documents
4. ✅ Extract entities and relationships using GPT-4o
5. ✅ Generate vector embeddings for all chunks
6. ✅ Create vector index for similarity search

**Expected Time**:
- Small dataset (10 PDFs, 100 records): ~5 minutes
- Medium dataset (100 PDFs, 1000 records): ~30 minutes
- Large dataset (1000+ PDFs, 10000+ records): ~2-4 hours

### 6. Query the Knowledge Graph

**Option A: Interactive Mode**
```bash
python main.py
# Select Option 8: Interactive query mode
```

**Example queries:**
```
🔍 Enter your question: What projects are managed by John Doe?

🔍 Enter your question: Find all documents that mention AACE procedures

🔍 Enter your question: What are the common risks mentioned in IFR documents?
```

**Option B: Programmatic Access**
```python
from src.rag.query_interface import GraphRAGQuery
from src.database.connection import get_connection

connection = get_connection()
query_interface = GraphRAGQuery(connection)

# Ask a question
answer = query_interface.answer_question(
    "What projects reference AACE procedures?"
)
print(answer)
```

## Verify Installation

### Test Database Connection
```bash
python -c "
from src.database.connection import get_connection
conn = get_connection()
print('✅ Neo4j connection successful!' if conn.verify_connection() else '❌ Connection failed')
"
```

### Check Ingested Data
```bash
python -c "
from src.database.connection import get_connection
conn = get_connection()
result = conn.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(n) as count')
for r in result:
    print(f'{r[\"type\"]}: {r[\"count\"]}')
"
```

Expected output:
```
Project: 15
Document: 50
Chunk: 1250
Person: 25
Team: 5
Procedure: 30
```

### Test Vector Search
```bash
python -c "
from src.rag.embeddings import EmbeddingManager
from src.database.connection import get_connection
conn = get_connection()
em = EmbeddingManager(conn)
results = em.similarity_search('AACE procedures', top_k=3)
print(f'✅ Found {len(results)} relevant chunks')
"
```

## Common Issues & Solutions

### Issue: "Failed to connect to Neo4j"
**Solution:**
```bash
# Check if Neo4j is running
docker ps | grep neo4j
# Or for local installation:
neo4j status

# Verify credentials in .env match Neo4j settings
# Check NEO4J_URI is correct (bolt://localhost:7687)
```

### Issue: "OpenAI API Error"
**Solution:**
```bash
# Verify API key
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('OPENAI_API_KEY')[:20] + '...')
"

# Check OpenAI account has credits
# Verify model name is correct: gpt-4o
```

### Issue: "No PDF files found"
**Solution:**
```bash
# Check PDF directory exists and contains files
ls -la data/pdfs/

# Ensure PDFs are readable
file data/pdfs/*.pdf
```

### Issue: "Out of memory during embedding generation"
**Solution:**
```env
# Reduce batch size in .env
BATCH_SIZE=50  # Instead of 100

# Or process in smaller chunks
# Run ingestion steps individually (Options 2-6)
```

### Issue: "Slow query performance"
**Solution:**
```python
# Reduce top_k parameter
query_interface.hybrid_query("your question", top_k=3)

# Increase score_threshold to get only high-quality matches
em.similarity_search("query", score_threshold=0.8)

# Check Neo4j indexes are created
python -c "
from src.database.schema import SchemaManager
from src.database.connection import get_connection
sm = SchemaManager(get_connection())
print(sm.get_schema_info())
"
```

## Next Steps

### 1. Explore the Knowledge Graph
```cypher
// Open Neo4j Browser at http://localhost:7474

// Visualize all relationships
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100

// Find most connected nodes
MATCH (n)
RETURN labels(n), n.name, count{(n)--()}  as degree
ORDER BY degree DESC
LIMIT 10
```

### 2. Customize Entity Extraction
Edit `src/extraction/entity_extractor.py` to add custom entity types:
```python
# Add new entity type: "Location"
# Modify extraction prompt to include:
- Location: Geographic locations or facilities
```

### 3. Add Custom Queries
Create custom query functions in `src/rag/query_interface.py`:
```python
def find_high_risk_projects(self) -> List[Dict[str, Any]]:
    """Find projects with multiple risk mentions."""
    query = """
    MATCH (proj:Project)<-[:GENERATED_FOR]-(d:Document)
    MATCH (d)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS_RISK]->(r:Risk)
    WITH proj, count(DISTINCT r) as risk_count
    WHERE risk_count > 5
    RETURN proj.name, risk_count
    ORDER BY risk_count DESC
    """
    return self.connection.execute_query(query)
```

### 4. Schedule Regular Updates
```bash
# Create cron job for daily ingestion
crontab -e

# Add line:
0 2 * * * cd /root/myprojects/Project_Document_Intelligence && python -c "from main import run_full_pipeline; run_full_pipeline()"
```

## Performance Tips

1. **Use GPU for embeddings**: Install CUDA-enabled sentence-transformers
2. **Increase Neo4j memory**: Edit `neo4j.conf` to allocate more RAM
3. **Batch processing**: Adjust `BATCH_SIZE` based on available memory
4. **Parallel processing**: Run entity extraction on multiple cores
5. **Cache embeddings**: Embeddings are stored in Neo4j, no need to regenerate

## Getting Help

1. Check logs: `tail -f logs/graphrag.log`
2. Review README.md for detailed documentation
3. Inspect code comments for implementation details
4. Use interactive Python shell to debug:
   ```python
   from src.database.connection import get_connection
   conn = get_connection()
   # Run test queries
   ```

---

**You're all set! Start querying your knowledge graph! 🚀**
