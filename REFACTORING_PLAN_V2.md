# GraphRAG v2: Enhanced PDF-Only Refactoring Plan
## Master Implementation Guide

**Document Version:** 2.0
**Date:** 2025-01-10
**Status:** APPROVED - Ready for Implementation
**Estimated Duration:** 2-3 weeks (6 phases)

---

## 🎯 Executive Summary

**Transformation Goal:** Convert dual-source GraphRAG (structured data + PDFs) into production-grade PDF-only system with enterprise reliability, quality tracking, and sophisticated query capabilities.

**Key Enhancements Over Original Plan:**
- ✅ **3-phase pipeline** (Validation → Extraction → Loading) with explicit error boundaries
- ✅ **Production-grade error handling** (DLQ, retry logic, model degradation)
- ✅ **Quality tracking system** (confidence scores, completeness metrics, prompt versioning)
- ✅ **Multi-modal query engine** (numeric, semantic, temporal search)
- ✅ **Comprehensive testing** (unit, integration, auto-generated regression tests)
- ✅ **Audit trail** (IngestionRun tracking, extraction metadata)

**Integration Sources:**
- Original Plan (10 tasks)
- Gemini Analysis (12 architectural improvements)
- Sequential Thinking Deep Analysis (6 missed opportunities identified)
- Total: 16 integrated enhancements, 8 Tier-1 (must-have), 4 Tier-2 (should-have), 4 Tier-3 (phase 2)

---

## 📊 Three-Phase Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: VALIDATION (Pre-Flight Checks)                │
├─────────────────────────────────────────────────────────┤
│  DocumentValidator:                                      │
│  ✓ PDF validity (not corrupted, OCR present)            │
│  ✓ File type (matches PAF pattern)                      │
│  ✓ File size (100KB < size < 50MB)                      │
│  └─> Invalid → ./data/rejected_files/ with metadata     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  PHASE 2: EXTRACTION (PDF → JSON)                       │
├─────────────────────────────────────────────────────────┤
│  PAFExtractor:                                           │
│  1. PyPDF text extraction                               │
│  2. LLM structured extraction (GPT-4o)                   │
│     ├─ Fields: project_number, title, manager, etc.     │
│     ├─ Confidence scores (1-10 per field)               │
│     └─ Prompt version tracking                          │
│  3. Retry logic: exponential backoff + model degradation│
│     (GPT-4o → GPT-4o-mini on failure)                   │
│  4. Save JSON: ./output/extracted_pafs/                 │
│  5. Auto-generate golden test file                      │
│  └─> Failures → ./data/failed_extractions/ with metadata│
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  PHASE 3: LOADING (JSON → Neo4j)                        │
├─────────────────────────────────────────────────────────┤
│  GraphLoader:                                            │
│  1. Pre-load validation (schema check, conflict detection)│
│  2. MERGE operations (upsert semantics)                 │
│     ├─ Create/update: Project, Person, Company          │
│     ├─ Create: Document, ScopeSummary, CostSummary      │
│     └─ Create: Revision with temporal chain             │
│  3. Generate embeddings (chunks + scope + cost)         │
│  4. Post-load integrity audits (Cypher validation queries)│
│  5. Track in IngestionRun node                          │
│  └─> Failures → ./data/failed_loads/ with metadata      │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Enhanced Neo4j Schema (v2)

### **Node Types**

#### **Core Entities**
```cypher
# Project (Core business entity)
(:Project {
  id: String [UNIQUE],
  name: String,
  confidence_score: Float,      # NEW: Average confidence across fields
  completeness_score: Float,    # NEW: % of fields successfully extracted
  prompt_version: String,        # NEW: Extraction prompt version
  created_at: DateTime,
  updated_at: DateTime
})

# Person (Managers, Sponsors, Initiators)
(:Person {
  name: String [UNIQUE]
})

# Company (Operating companies)
(:Company {
  name: String [UNIQUE]
})
```

#### **Document-Related Nodes**
```cypher
# Document (PAF files)
(:Document {
  id: String [UNIQUE],
  type: String,                  # "PAF", "SRF", "IFR"
  path: String,
  file_name: String,
  file_hash: String,             # NEW: SHA256 for duplicate detection
  num_pages: Integer,
  prompt_version: String,        # NEW: Extraction prompt version
  extraction_model: String,      # NEW: "gpt-4o" or "gpt-4o-mini"
  extraction_timestamp: DateTime # NEW: When extracted
})

# ScopeSummary (Full text of [3] Project Scope section)
(:ScopeSummary {
  id: String [UNIQUE],
  text: String,
  embedding: Vector,             # For semantic search
  confidence_score: Float,       # NEW: LLM confidence in extraction
  prompt_version: String         # NEW: Extraction prompt version
})

# CostSummary (Full text of [5] Cost and Funding section + financials)
(:CostSummary {
  id: String [UNIQUE],
  text: String,
  embedding: Vector,             # For semantic search
  confidence_score: Float,       # NEW: LLM confidence in extraction
  total_request: Float,
  year_1_cost: Float,            # NEW: Enhanced financial breakdown
  year_2_cost: Float,            # NEW
  year_3_cost: Float,            # NEW
  contingency_percent: Float,    # NEW
  funding_source: String,        # NEW: "internal", "external", "mixed"
  prompt_version: String         # NEW
})

# Revision (Document version tracking)
(:Revision {
  id: String [UNIQUE],
  version: String,               # "Rev 1", "Rev 2", etc.
  date: Date,
  description: String
})

# Chunk (Text chunks for vector search)
(:Chunk {
  chunk_id: String [UNIQUE],
  text: String,
  embedding: Vector,
  chunk_index: Integer,
  start_char: Integer,
  end_char: Integer
})
```

#### **Tracking & Audit Nodes**
```cypher
# IngestionRun (Pipeline execution tracking)
(:IngestionRun {
  run_id: String [UNIQUE],
  timestamp: DateTime,
  files_processed: Integer,
  files_failed: Integer,
  files_rejected: Integer,       # NEW: Pre-flight rejections
  prompt_version: String,
  duration: Float,               # Seconds
  success_rate: Float            # files_processed / (processed + failed)
})
```

### **Relationships**

```cypher
# Project relationships
(Person)-[:MANAGES]->(Project)
(Person)-[:SPONSORS]->(Project)
(Person)-[:INITIATED]->(Project)
(Project)-[:HAS_DOCUMENT]->(Document)
(Project)-[:HAS_SCOPE]->(ScopeSummary)
(Project)-[:HAS_COST]->(CostSummary)
(Project)-[:OPERATED_BY]->(Company)

# Document relationships
(Document)-[:HAS_CHUNK]->(Chunk)
(Document)-[:HAS_REVISION]->(Revision)

# Revision temporal chain (NEW)
(Revision)-[:NEXT_REVISION]->(Revision)

# Ingestion tracking (NEW)
(IngestionRun)-[:PROCESSED]->(Document)
(IngestionRun)-[:FAILED {reason: String}]->(Document)
(IngestionRun)-[:REJECTED {reason: String}]->(Document)

# Future: Review tracking (placeholder)
# (Revision)-[:REVIEWED_BY]->(Person)
```

### **Constraints & Indexes**

```cypher
# Uniqueness constraints
CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT company_name_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT scope_id_unique IF NOT EXISTS FOR (s:ScopeSummary) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT cost_id_unique IF NOT EXISTS FOR (cs:CostSummary) REQUIRE cs.id IS UNIQUE;
CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (r:Revision) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT ingestion_run_id_unique IF NOT EXISTS FOR (ir:IngestionRun) REQUIRE ir.run_id IS UNIQUE;

# Property indexes
CREATE INDEX document_type_paf_index IF NOT EXISTS FOR (d:Document) ON (d.type);
CREATE INDEX revision_date_index IF NOT EXISTS FOR (r:Revision) ON (r.date);
CREATE INDEX project_confidence_index IF NOT EXISTS FOR (p:Project) ON (p.confidence_score);
CREATE INDEX cost_total_index IF NOT EXISTS FOR (cs:CostSummary) ON (cs.total_request);
CREATE INDEX document_hash_index IF NOT EXISTS FOR (d:Document) ON (d.file_hash);

# Vector indexes
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX scope_embeddings IF NOT EXISTS
FOR (s:ScopeSummary) ON s.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX cost_embeddings IF NOT EXISTS
FOR (cs:CostSummary) ON cs.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
```

---

## 📦 Enhanced Data Models (Pydantic)

### **Core PAF Document Model**

```python
# src/models/paf_document.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class Revision(BaseModel):
    """Document revision metadata."""
    version: str
    date: Optional[str] = None
    description: Optional[str] = None

class ConfidenceScores(BaseModel):
    """Confidence scores for extracted fields (1-10 scale)."""
    project_number: int = Field(ge=1, le=10)
    project_title: int = Field(ge=1, le=10)
    company: int = Field(ge=1, le=10)
    project_manager: int = Field(ge=1, le=10)
    project_sponsor: int = Field(ge=1, le=10)
    project_initiator: int = Field(ge=1, le=10)
    scope_text: int = Field(ge=1, le=10)
    cost_text: int = Field(ge=1, le=10)

class CostBreakdown(BaseModel):
    """Enhanced financial metadata."""
    total_request: Optional[float] = None
    year_1_cost: Optional[float] = None
    year_2_cost: Optional[float] = None
    year_3_cost: Optional[float] = None
    contingency_percent: Optional[float] = None
    funding_source: Optional[str] = None  # "internal", "external", "mixed"

class PAFDocument(BaseModel):
    """Complete PAF document extraction result."""
    # Header fields (REQUIRED)
    project_number: str
    project_title: str
    company: str

    # Header fields (OPTIONAL)
    project_manager: Optional[str] = None
    project_sponsor: Optional[str] = None
    project_initiator: Optional[str] = None

    # Content sections (REQUIRED)
    scope_text: str  # Full [3] Project Scope section
    cost_text: str   # Full [5] Cost and Funding section

    # Financial data
    cost_breakdown: CostBreakdown

    # Revisions
    revisions: List[Revision] = []

    # Quality metadata
    confidence_scores: ConfidenceScores
    completeness_score: float = Field(ge=0.0, le=1.0, description="Fraction of fields successfully extracted")

    # Extraction metadata
    prompt_version: str
    extraction_model: str  # "gpt-4o" or "gpt-4o-mini"
    extraction_timestamp: datetime
    file_hash: str

    def calculate_average_confidence(self) -> float:
        """Calculate average confidence across all fields."""
        scores = self.confidence_scores.dict().values()
        return sum(scores) / len(scores)

    def is_required_fields_complete(self) -> bool:
        """Check if all required fields are present."""
        return bool(
            self.project_number and
            self.project_title and
            self.company and
            self.scope_text and
            self.cost_text
        )
```

---

## 🔨 Implementation Plan (6 Phases)

### **PHASE 1: Foundation & Cleanup**
**Duration:** 2-3 days
**Risk:** LOW

#### **Task 1.1: Update Dependencies**
**File:** `requirements.txt`

```diff
# Remove
- duckdb>=0.9.0
- openpyxl>=3.1.0

# Add (optional for tooling)
+ streamlit>=1.30.0  # Optional: For visualizer tool
```

#### **Task 1.2: Create Directory Structure**
```bash
mkdir -p data/rejected_files
mkdir -p data/failed_extractions
mkdir -p data/failed_loads
mkdir -p output/extracted_pafs
mkdir -p tests/golden_files
mkdir -p prompts
```

#### **Task 1.3: Delete Obsolete Files**
```bash
rm src/ingestion/structured_data.py
# Verify no other files import this module
```

#### **Task 1.4: Update Configuration**
**File:** `.env.example`

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_FALLBACK_MODEL=gpt-4o-mini

# Data Directories (UPDATED)
PDF_DATA_DIR=./data/pdfs
OUTPUT_DIR=./output
REJECTED_FILES_DIR=./data/rejected_files
FAILED_EXTRACTIONS_DIR=./data/failed_extractions
FAILED_LOADS_DIR=./data/failed_loads

# Processing Configuration
BATCH_SIZE=100
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384

# NEW: Extraction Configuration
EXTRACTION_PROMPT_VERSION=v1.0
ENABLE_MODEL_DEGRADATION=true
RETRY_MAX_ATTEMPTS=3
RETRY_BACKOFF_BASE=2  # Seconds: 2, 4, 8...

# NEW: Quality Thresholds
CONFIDENCE_THRESHOLD=6.0
COMPLETENESS_THRESHOLD=0.8
AUTO_GENERATE_GOLDEN_TESTS=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/graphrag.log
```

#### **Task 1.5: Create Prompt Templates Directory**
**File:** `prompts/v1.0_paf_extraction.txt`

```
You are analyzing a Project Authorization Form (PAF) document. Extract information with HIGH PRECISION and provide CONFIDENCE SCORES (1-10 scale) for each field.

**HEADER FIELDS [REQUIRED]:**
- Project Number: Look for "Project Number", "Project ID", or similar field at document top
- Project Title: Main project name/title (usually bold or prominent)
- Company: Operating company name (e.g., "Eversource NH", "NSTAR Electric")

**HEADER FIELDS [OPTIONAL]:**
- Project Manager: Name of person managing project (look for "Project Manager" label)
- Project Sponsor: Name of executive sponsor (look for "Sponsor" or "Executive Sponsor")
- Project Initiator: Person who initiated the project (look for "Initiator" or "Requested By")

**FINANCIAL DATA (Extract from [5] Cost and Funding section):**
- Total Capital Request: Total dollar amount requested (look for "Total Request", "Total Cost", "Capital Request")
- Year 1 Cost: First year expenditure
- Year 2 Cost: Second year expenditure
- Year 3 Cost: Third year expenditure
- Contingency Percent: Contingency as percentage (e.g., "15% contingency")
- Funding Source: "internal", "external", or "mixed"

**CONTENT SECTIONS [REQUIRED]:**
- Section [3] Project Scope: Extract the ENTIRE text of section 3, including all subsections and bullets
- Section [5] Cost and Funding: Extract the ENTIRE text of section 5, including all financial tables

**REVISION HISTORY:**
Extract all revisions from revision history table. For each revision, extract:
- Version: "Rev 1", "Rev 2", "Rev A", etc.
- Date: Revision date (YYYY-MM-DD format preferred)
- Description: Brief description of changes

**CONFIDENCE SCORING INSTRUCTIONS:**
For each field, provide a confidence score 1-10 where:
- 10: Field explicitly present with clear label, no ambiguity
- 7-9: Field present but label unclear or value requires interpretation
- 4-6: Field possibly present but significant uncertainty
- 1-3: Field not found or highly ambiguous

**OUTPUT FORMAT:**
Return valid JSON matching this exact structure:
{
  "project_number": "string",
  "project_title": "string",
  "company": "string",
  "project_manager": "string or null",
  "project_sponsor": "string or null",
  "project_initiator": "string or null",
  "scope_text": "full text of section 3",
  "cost_text": "full text of section 5",
  "cost_breakdown": {
    "total_request": float or null,
    "year_1_cost": float or null,
    "year_2_cost": float or null,
    "year_3_cost": float or null,
    "contingency_percent": float or null,
    "funding_source": "string or null"
  },
  "revisions": [
    {"version": "string", "date": "string or null", "description": "string or null"}
  ],
  "confidence_scores": {
    "project_number": int (1-10),
    "project_title": int (1-10),
    "company": int (1-10),
    "project_manager": int (1-10),
    "project_sponsor": int (1-10),
    "project_initiator": int (1-10),
    "scope_text": int (1-10),
    "cost_text": int (1-10)
  }
}

**IMPORTANT:**
- Return ONLY valid JSON, no additional text
- If a field is not found, use null (not empty string)
- Be conservative with confidence scores - uncertainty is acceptable
```

---

### **PHASE 2: Enhanced Schema & Data Models**
**Duration:** 2-3 days
**Risk:** MEDIUM

#### **Task 2.1: Update Schema Manager**
**File:** `src/database/schema.py`

**Implementation Notes:**
- Add all new constraints listed in schema section above
- Create helper method `create_vector_index(index_name, node_label, dimension)`
- Remove old constraints for Team, Procedure, Deliverable
- Add comprehensive `get_schema_info()` to verify all constraints/indexes created

#### **Task 2.2: Create PAF Document Models**
**File:** `src/models/__init__.py` (new directory)
**File:** `src/models/paf_document.py`

**Implementation Notes:**
- Implement complete Pydantic models as shown above
- Add validation methods: `is_required_fields_complete()`, `calculate_average_confidence()`
- Add JSON serialization helpers for Neo4j compatibility

#### **Task 2.3: Create Validation Models**
**File:** `src/models/validation.py`

```python
from pydantic import BaseModel
from enum import Enum

class ValidationStatus(Enum):
    VALID = "valid"
    REJECTED = "rejected"
    WARNING = "warning"

class ValidationResult(BaseModel):
    status: ValidationStatus
    reasons: List[str]
    file_path: str
    file_size: int
    file_hash: str

class IntegrityViolation(BaseModel):
    violation_type: str  # "orphaned_project", "missing_cost", etc.
    entity_id: str
    description: str
    severity: str  # "error", "warning"
```

---

### **PHASE 3: Validation & Extraction Pipeline**
**Duration:** 4-5 days
**Risk:** HIGH (LLM integration)

#### **Task 3.1: Document Validator**
**File:** `src/validation/document_validator.py`

**Key Methods:**
```python
class DocumentValidator:
    def validate_pdf(self, file_path: Path) -> ValidationResult:
        """
        Pre-flight validation checks:
        1. File exists and is readable
        2. File is valid PDF (pypdf can open)
        3. File name matches PAF pattern (regex: .*PAF.*\.pdf)
        4. File size within bounds (100KB < size < 50MB)
        5. PDF has text layer (OCR check)
        """

    def check_ocr_present(self, file_path: Path) -> bool:
        """Check if PDF has extractable text (not scanned image)."""

    def move_to_rejected(self, file_path: Path, reason: str):
        """Move to rejected_files/ with metadata JSON."""
```

**Error Handling:**
- Catch all exceptions, log detailed error info
- Create rejection metadata: `{filename, rejection_reason, timestamp, file_size, attempted_at}`

#### **Task 3.2: PAF Extractor Core**
**File:** `src/extraction/paf_extractor.py`

**Key Methods:**
```python
class PAFExtractor:
    def __init__(self, connection: Optional[Neo4jConnection] = None):
        self.llm_primary = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt_version = get_settings().extraction.prompt_version
        self.prompt_template = self._load_prompt_template()

    def extract_with_retry(self, file_path: Path) -> PAFDocument:
        """
        Main extraction with retry logic.

        Retry Strategy:
        1. Attempt 1: GPT-4o (primary)
        2. Wait 2s, Attempt 2: GPT-4o
        3. Wait 4s, Attempt 3: GPT-4o
        4. If all fail, Attempt 4: GPT-4o-mini (fallback)
        5. If still fails: raise ExtractionError → move to DLQ
        """

    def _extract_once(self, text: str, model: ChatOpenAI) -> PAFDocument:
        """Single extraction attempt with specific model."""

    def calculate_completeness_score(self, paf_doc: PAFDocument) -> float:
        """
        Completeness = (fields_present / total_fields)

        Required fields (weight 1.0): project_number, project_title, company, scope_text, cost_text
        Optional fields (weight 0.5): manager, sponsor, initiator, financial breakdown
        """

    def save_json(self, paf_doc: PAFDocument, output_dir: Path) -> Path:
        """Save extracted JSON to output/extracted_pafs/{project_number}.json"""

    def generate_golden_test(self, paf_doc: PAFDocument, pdf_path: Path):
        """Auto-generate pytest test in tests/golden_files/"""
```

**Error Handling:**
- Wrap each extraction attempt in try-except
- Log all attempts with model name, timestamp, token usage
- On final failure, save partial extraction to failed_extractions/ with full error context

#### **Task 3.3: LLM Prompt Manager**
**File:** `src/extraction/prompt_manager.py`

```python
class PromptManager:
    def load_prompt(self, version: str) -> str:
        """Load prompt template from prompts/v{version}_paf_extraction.txt"""

    def get_current_version(self) -> str:
        """Get current prompt version from config"""

    def list_available_versions(self) -> List[str]:
        """List all prompt versions in prompts/ directory"""
```

#### **Task 3.4: Golden Test Generator**
**File:** `src/testing/golden_test_generator.py`

```python
class GoldenTestGenerator:
    def generate_test(self, paf_doc: PAFDocument, pdf_path: Path):
        """
        Create test file: tests/golden_files/test_golden_{project_number}.py

        Test structure:
        - Save PDF to tests/fixtures/
        - Save JSON to tests/golden_files/{project_number}_golden.json
        - Generate pytest test that re-extracts and compares
        """
```

---

### **PHASE 4: Loading Pipeline & Integrity**
**Duration:** 3-4 days
**Risk:** MEDIUM

#### **Task 4.1: Graph Loader Core**
**File:** `src/loading/graph_loader.py`

**Key Methods:**
```python
class GraphLoader:
    def load_from_json(self, json_path: Path) -> LoadResult:
        """
        Complete loading pipeline:
        1. Load and validate JSON
        2. Pre-load validation checks
        3. Build graph structure (MERGE operations)
        4. Generate embeddings
        5. Post-load integrity audits
        6. Track in IngestionRun
        """

    def pre_load_validation(self, paf_doc: PAFDocument) -> List[ValidationError]:
        """
        Pre-load checks:
        - Duplicate project_number detection
        - Required fields present
        - Completeness score > threshold
        - Confidence scores reasonable (not all zeros)
        """

    def build_graph_structure(self, paf_doc: PAFDocument):
        """
        Execute MERGE operations for all nodes and relationships.

        Order matters:
        1. MERGE Project
        2. MERGE Person nodes (Manager, Sponsor, Initiator)
        3. MERGE Company
        4. MERGE Document
        5. MERGE ScopeSummary
        6. MERGE CostSummary
        7. MERGE Revisions (with NEXT_REVISION chain)
        8. CREATE relationships
        """

    def create_revision_chain(self, revisions: List[Revision], document_id: str):
        """
        Create temporal chain: Rev1→Rev2→Rev3
        Sort by date, create NEXT_REVISION relationships
        """

    def post_load_integrity_audits(self) -> List[IntegrityViolation]:
        """
        Run Cypher validation queries:
        1. Orphaned Projects (no Scope)
        2. Missing Costs
        3. Duplicate Projects
        4. Invalid totals (cost <= 0)
        5. Low confidence records
        """
```

#### **Task 4.2: Integrity Audit Manager**
**File:** `src/loading/integrity_audits.py`

```python
class IntegrityAuditManager:
    AUDIT_QUERIES = {
        "orphaned_projects": """
            MATCH (p:Project)
            WHERE NOT EXISTS ((p)-[:HAS_SCOPE]->())
            RETURN p.id as project_id, p.name as project_name
        """,
        "missing_costs": """
            MATCH (p:Project)
            WHERE NOT EXISTS ((p)-[:HAS_COST]->())
            RETURN p.id as project_id
        """,
        "duplicate_projects": """
            MATCH (p:Project)
            WITH p.id as pid, count(p) as cnt
            WHERE cnt > 1
            RETURN pid, cnt
        """,
        "invalid_totals": """
            MATCH (cs:CostSummary)
            WHERE cs.total_request <= 0 OR cs.total_request IS NULL
            RETURN cs.id as cost_id
        """,
        "low_confidence": """
            MATCH (n)
            WHERE n.confidence_score IS NOT NULL AND n.confidence_score < 6.0
            RETURN labels(n)[0] as node_type, id(n) as node_id, n.confidence_score as score
        """
    }

    def run_all_audits(self) -> AuditReport:
        """Execute all audits, return consolidated report"""

    def run_audit(self, audit_name: str) -> List[IntegrityViolation]:
        """Execute single audit query"""
```

#### **Task 4.3: Ingestion Run Tracker**
**File:** `src/loading/ingestion_tracker.py`

```python
class IngestionRunTracker:
    def start_run(self) -> str:
        """
        Create IngestionRun node with:
        - run_id (UUID)
        - timestamp
        - prompt_version
        - status: "in_progress"

        Returns: run_id
        """

    def track_success(self, run_id: str, document_id: str):
        """Create (IngestionRun)-[:PROCESSED]->(Document)"""

    def track_failure(self, run_id: str, document_id: str, reason: str):
        """Create (IngestionRun)-[:FAILED {reason}]->(Document)"""

    def complete_run(self, run_id: str, stats: Dict):
        """
        Update IngestionRun with final stats:
        - files_processed
        - files_failed
        - duration
        - success_rate
        - status: "completed"
        """
```

#### **Task 4.4: Update Embedding Manager**
**File:** `src/rag/embeddings.py`

**New Methods:**
```python
class EmbeddingManager:
    def generate_scope_embeddings(self) -> int:
        """
        Generate embeddings for ScopeSummary nodes.

        Query: MATCH (s:ScopeSummary) WHERE s.embedding IS NULL
        Process: Generate embeddings for s.text
        Update: SET s.embedding = embedding_vector
        Returns: Number of embeddings generated
        """

    def generate_cost_embeddings(self) -> int:
        """
        Generate embeddings for CostSummary nodes.

        Query: MATCH (cs:CostSummary) WHERE cs.embedding IS NULL
        Process: Generate embeddings for cs.text
        Update: SET cs.embedding = embedding_vector
        Returns: Number of embeddings generated
        """

    def run_full_embedding_pipeline(self):
        """
        UPDATED: Generate all embeddings.

        Order:
        1. Chunk embeddings (existing)
        2. ScopeSummary embeddings (NEW)
        3. CostSummary embeddings (NEW)

        Uses batch processing for efficiency.
        """
```

---

### **PHASE 5: Enhanced Query Engine**
**Duration:** 3-4 days
**Risk:** MEDIUM

#### **Task 5.1: Query Classifier**
**File:** `src/query/query_classifier.py`

```python
from enum import Enum
import re

class QueryType(Enum):
    FINANCIAL = "financial"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"

class NumericFilter(BaseModel):
    field: str  # "total_request"
    operator: str  # ">", "<", ">=", "<=", "="
    value: float

class QueryClassifier:
    FINANCIAL_PATTERNS = [
        r'\$\d+[KMB]?',
        r'cost.*over',
        r'cost.*under',
        r'budget.*greater',
        r'total.*request',
    ]

    TEMPORAL_PATTERNS = [
        r'\d{4}',  # Year
        r'Q[1-4]\s*\d{4}',  # Quarter
        r'(january|february|...|december)',
        r'last\s+(year|quarter|month)',
    ]

    def classify(self, question: str) -> QueryType:
        """
        Analyze question to determine query type.

        Logic:
        1. Check for financial patterns → FINANCIAL
        2. Check for temporal patterns → TEMPORAL
        3. If both → HYBRID
        4. Default → SEMANTIC
        """

    def extract_numeric_predicate(self, question: str) -> Optional[NumericFilter]:
        """
        Extract numeric filter from financial queries.

        Examples:
        - "projects over $2M" → NumericFilter(field='total_request', op='>', value=2000000)
        - "costs under $500K" → NumericFilter(field='total_request', op='<', value=500000)
        - "budget of $1.5 million" → NumericFilter(field='total_request', op='=', value=1500000)

        Handles: K (thousands), M (millions), B (billions)
        """
```

#### **Task 5.2: Multi-Modal Query Engine**
**File:** `src/query_engine.py` (renamed from `src/rag/query_interface.py`)

```python
class GraphRAGQueryEngine:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
        self.embedding_manager = EmbeddingManager(connection)
        self.query_classifier = QueryClassifier()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def hybrid_query(self, question: str, top_k: int = 5) -> QueryResult:
        """
        Main query interface with intelligent routing.

        Workflow:
        1. Classify query type
        2. Execute appropriate search(es)
        3. Merge and rank results
        4. Generate LLM answer with context
        """

    def financial_search(
        self,
        question: str,
        numeric_filter: NumericFilter,
        top_k: int
    ) -> List[Result]:
        """
        Cypher property search + optional vector search.

        Strategy:
        1. Execute Cypher filter on CostSummary.total_request
        2. If semantic terms present, also do vector search
        3. Merge results, prioritize Cypher matches
        """

    def semantic_search(self, question: str, top_k: int) -> List[Result]:
        """
        Vector search across multiple node types.

        Strategy:
        1. Search ScopeSummary embeddings (weight: 0.4)
        2. Search CostSummary embeddings (weight: 0.3)
        3. Search Chunk embeddings (weight: 0.3)
        4. Merge with weighted scoring
        """

    def temporal_search(
        self,
        question: str,
        date_filter: DateFilter,
        top_k: int
    ) -> List[Result]:
        """
        Date-based Cypher queries on Revision.date.

        Example:
        MATCH (d:Document)-[:HAS_REVISION]->(r:Revision)
        WHERE r.date >= date('2023-01-01') AND r.date <= date('2023-12-31')
        RETURN d, r
        """

    def graph_context_expansion(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """
        UPDATED: Expand context with new node types.

        Query additions:
        OPTIONAL MATCH (proj)-[:HAS_SCOPE]->(scope:ScopeSummary)
        OPTIONAL MATCH (proj)-[:HAS_COST]->(cost:CostSummary)
        OPTIONAL MATCH (proj)-[:OPERATED_BY]->(comp:Company)
        OPTIONAL MATCH (d)-[:HAS_REVISION]->(rev:Revision)

        Returns enhanced context with:
        - scope_summary (text)
        - cost_summary (text)
        - total_cost (float)
        - company (name)
        - revisions (list)
        """

    def _generate_answer(
        self,
        question: str,
        results: List[Result],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate LLM answer with enhanced context.

        Prompt updates:
        - Include scope/cost summaries when available
        - Format financial data ($2.5M instead of 2500000)
        - Cite specific sections ([3] Project Scope)
        - Include revision history context
        """
```

---

### **PHASE 6: Integration, Testing & Migration**
**Duration:** 3-4 days
**Risk:** LOW-MEDIUM

#### **Task 6.1: Update Main Orchestrator**
**File:** `main.py`

**Major Changes:**
```python
# REMOVE IMPORTS
# from src.ingestion.structured_data import StructuredDataIngestion

# ADD IMPORTS
from src.validation.document_validator import DocumentValidator
from src.extraction.paf_extractor import PAFExtractor
from src.loading.graph_loader import GraphLoader
from src.loading.ingestion_tracker import IngestionRunTracker
from src.query_engine import GraphRAGQueryEngine

def run_full_pipeline() -> None:
    """
    Execute v2 three-phase pipeline.

    Phases:
    1. Initialize schema
    2. Validate documents
    3. Extract PAF data
    4. Load into Neo4j
    5. Generate embeddings
    6. Run integrity audits
    """
    logger.info("Starting GraphRAG v2 Pipeline")

    # Create ingestion run tracker
    tracker = IngestionRunTracker(connection)
    run_id = tracker.start_run()

    try:
        # Phase 1: Validation
        validated_files = validate_documents()

        # Phase 2: Extraction
        extracted_files = extract_paf_documents(validated_files, run_id, tracker)

        # Phase 3: Loading
        load_extracted_documents(extracted_files, run_id, tracker)

        # Phase 4: Embeddings
        generate_embeddings()

        # Phase 5: Audits
        run_integrity_audits()

        tracker.complete_run(run_id, stats)
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        tracker.fail_run(run_id, str(e))
        raise

def validate_documents() -> List[Path]:
    """Phase 1: Validate all PDFs."""
    validator = DocumentValidator()
    pdf_files = list(settings.data.pdf_data_dir.glob("**/*.pdf"))

    validated = []
    for pdf_file in tqdm(pdf_files, desc="Validating PDFs"):
        result = validator.validate_pdf(pdf_file)
        if result.status == ValidationStatus.VALID:
            validated.append(pdf_file)
        else:
            validator.move_to_rejected(pdf_file, result.reasons)

    return validated

def extract_paf_documents(
    pdf_files: List[Path],
    run_id: str,
    tracker: IngestionRunTracker
) -> List[Path]:
    """Phase 2: Extract PAF data to JSON."""
    extractor = PAFExtractor()
    extracted = []

    for pdf_file in tqdm(pdf_files, desc="Extracting PAFs"):
        try:
            paf_doc = extractor.extract_with_retry(pdf_file)
            json_path = extractor.save_json(paf_doc, settings.output_dir)

            if settings.auto_generate_golden_tests:
                extractor.generate_golden_test(paf_doc, pdf_file)

            extracted.append(json_path)

        except ExtractionError as e:
            logger.error(f"Extraction failed for {pdf_file.name}: {e}")
            tracker.track_failure(run_id, pdf_file.name, str(e))

    return extracted

def load_extracted_documents(
    json_files: List[Path],
    run_id: str,
    tracker: IngestionRunTracker
):
    """Phase 3: Load JSONs into Neo4j."""
    loader = GraphLoader(connection)

    for json_path in tqdm(json_files, desc="Loading into Neo4j"):
        try:
            result = loader.load_from_json(json_path)
            tracker.track_success(run_id, result.document_id)

        except LoadError as e:
            logger.error(f"Load failed for {json_path.name}: {e}")
            tracker.track_failure(run_id, json_path.name, str(e))

def run_integrity_audits():
    """Phase 5: Run post-load integrity audits."""
    from src.loading.integrity_audits import IntegrityAuditManager

    auditor = IntegrityAuditManager(connection)
    report = auditor.run_all_audits()

    if report.has_errors():
        logger.warning(f"Integrity violations found: {report.error_count}")
        logger.warning(report.summary())
    else:
        logger.info("All integrity audits passed")
```

**Updated Menu:**
```python
print("\n" + "=" * 80)
print("GraphRAG Knowledge System v2 (PDF-Only)")
print("=" * 80)
print("\nOptions:")
print("1. Run full ingestion pipeline (3-phase: validate → extract → load)")
print("2. Initialize database schema only")
print("3. Validate documents only")
print("4. Extract PAF documents only")
print("5. Load extracted documents only")
print("6. Generate embeddings only")
print("7. Run integrity audits")
print("8. Run example queries")
print("9. Interactive query mode")
print("10. View ingestion run history")
print("11. Exit")
print("=" * 80)
```

#### **Task 6.2: Comprehensive Test Suite**

**Test Structure:**
```
tests/
├── unit/
│   ├── test_document_validator.py
│   ├── test_paf_extractor.py
│   ├── test_graph_loader.py
│   ├── test_query_classifier.py
│   └── test_integrity_audits.py
├── integration/
│   ├── test_extraction_pipeline.py
│   ├── test_loading_pipeline.py
│   └── test_query_engine.py
├── golden/
│   ├── test_golden_files.py (auto-generated)
│   └── fixtures/
│       ├── sample_paf.pdf
│       └── sample_paf_golden.json
└── performance/
    └── test_load_performance.py
```

**Key Test Files:**

**tests/unit/test_paf_extractor.py:**
```python
def test_extraction_success(mock_llm):
    """Test successful extraction with valid PAF."""

def test_extraction_retry_logic(mock_llm_with_failures):
    """Test retry with exponential backoff."""

def test_model_degradation(mock_primary_fails, mock_fallback_succeeds):
    """Test fallback to GPT-4o-mini."""

def test_confidence_score_calculation():
    """Test confidence score aggregation."""

def test_completeness_score_required_fields():
    """Test completeness calculation with required fields."""

def test_completeness_score_optional_fields():
    """Test completeness with optional fields missing."""
```

**tests/integration/test_extraction_pipeline.py:**
```python
def test_end_to_end_extraction_with_real_llm():
    """Integration test with actual LLM (requires API key)."""

def test_golden_file_generation():
    """Test that golden test files are created correctly."""
```

**Coverage Target:** >80% for core pipeline (validation, extraction, loading)

#### **Task 6.3: Migration Scripts**

**File:** `scripts/migrate_v1_to_v2.py`
```python
#!/usr/bin/env python3
"""
Migrate existing v1 database to v2 schema.

Usage:
    python scripts/migrate_v1_to_v2.py --backup --dry-run
    python scripts/migrate_v1_to_v2.py --execute
"""

def migrate_v1_to_v2(backup_first: bool = True, dry_run: bool = False):
    """
    Migration steps:
    1. Backup v1 data (if backup_first=True)
       - Export all nodes and relationships to JSON
       - Save to backups/v1_backup_{timestamp}.json

    2. Preserve compatible data
       - Export Person nodes → preserve
       - Export Project nodes → preserve
       - Export Document nodes → preserve (update schema)

    3. Remove v1-specific nodes
       - DROP Team nodes and relationships
       - DROP Procedure nodes and relationships
       - DROP Deliverable nodes and relationships

    4. Add v2 schema elements
       - CREATE new constraints (Company, ScopeSummary, etc.)
       - CREATE new indexes
       - CREATE vector indexes

    5. Update existing nodes
       - Add new properties (confidence_score, completeness_score, prompt_version)
       - Set default values for missing properties

    6. Verify migration
       - Run integrity audits
       - Check constraint violations
       - Verify vector indexes created

    7. Generate migration report
       - Nodes migrated/deleted/created
       - Relationships migrated/deleted/created
       - Any warnings or errors
    """
```

**File:** `scripts/rollback_v2_to_v1.py`
```python
def rollback_to_v1(backup_file: Path):
    """
    Rollback to v1 schema from backup.

    Steps:
    1. Load backup JSON
    2. Clear current database
    3. Recreate v1 schema
    4. Restore nodes and relationships
    5. Verify restoration
    """
```

#### **Task 6.4: Update Documentation**

**README.md:** Update all sections as outlined in revised plan
**QUICKSTART.md:** Remove structured data references, add v2 pipeline examples
**API.md:** (NEW) Document query engine API with examples

---

## 🎯 Implementation Priority Matrix

### **TIER 1: Must-Have (Implement Immediately)**
✅ **HIGH VALUE, LOW-MEDIUM EFFORT**

1. **Extractor/Loader Separation** - Enables independent testing, retry logic
2. **Pre-flight Validation** - Saves costs, prevents bad data early
3. **Dead Letter Queue** - Essential error handling for production
4. **LLM Confidence Scores** - Quality tracking and manual review queue
5. **Post-load Integrity Audits** - Prevents silent data corruption
6. **Automated Test Generation** - Builds regression suite organically
7. **Prompt Versioning** - Data provenance, selective re-extraction
8. **Partial Extraction Recovery** - Maximize data utilization

**Estimated Effort:** 10-12 days
**Risk:** LOW-MEDIUM

---

### **TIER 2: Should-Have (Implement Second)**
✅ **HIGH VALUE, MEDIUM-HIGH EFFORT**

9. **Hybrid Query Engine** - Critical for financial queries
10. **Revision Chain Modeling** - Temporal tracking foundation
11. **Ingestion Run Tracking** - Audit trail and monitoring
12. **Enhanced Financial Metadata** - Sophisticated financial queries

**Estimated Effort:** 4-5 days
**Risk:** MEDIUM

---

### **TIER 3: Nice-to-Have (Phase 2)**
⏳ **DEFER TO FUTURE**

13. **Semantic Caching (Redis)** - Cost optimization (~60% savings)
14. **Section-Aware Chunking** - Improved search signal-to-noise
15. **Full Staging Graph** - Enterprise-grade data validation
16. **Single-Document Visualizer** - Debugging/QA tool (Streamlit)

**Estimated Effort:** 5-7 days
**Risk:** MEDIUM-HIGH

---

## ⚠️ Risk Assessment & Mitigation

### **Risk 1: LLM Extraction Accuracy**
**Probability:** MEDIUM | **Impact:** HIGH

**Scenario:** PAF format variations confuse LLM, leading to incorrect or incomplete extractions.

**Mitigation Strategy:**
- ✅ Confidence scores flag uncertain extractions
- ✅ Completeness scores track missing fields
- ✅ Golden file regression tests catch accuracy regressions
- ✅ Manual review queue for low-confidence extractions (<6.0)
- ✅ Prompt versioning enables iterative improvements

**Monitoring:** Track average confidence score (target: >7.0), extraction success rate (target: >90%)

---

### **Risk 2: Performance Bottleneck**
**Probability:** MEDIUM | **Impact:** MEDIUM

**Scenario:** Dual-pass extraction + embeddings = 3-5 sec/PDF, total 3-4 hours for 3,500 PDFs.

**Mitigation Strategy:**
- ✅ Async processing (parallel extraction)
- ✅ Batch operations for Neo4j writes
- ✅ JSON caching prevents re-extraction
- ✅ Model degradation reduces retry costs

**Monitoring:** Track extraction time (target: <3 sec/PDF), full pipeline time (target: <3 hours)

---

### **Risk 3: LLM Cost Overruns**
**Probability:** LOW | **Impact:** MEDIUM

**Scenario:** 3,500 PDFs × $0.05/extraction = $175/run. Multiple re-runs = $500+.

**Mitigation Strategy:**
- ✅ Retry with cheaper model (GPT-4o-mini 60% cheaper)
- ✅ Cache extracted JSONs for reprocessing
- ✅ Only re-extract on prompt version changes
- ✅ Track extraction costs per document

**Monitoring:** Track cost per extraction (target: <$0.05), failed extraction rate (target: <5%)

---

### **Risk 4: Data Quality Issues**
**Probability:** MEDIUM | **Impact:** HIGH

**Scenario:** Incomplete PAFs, malformed documents, missing sections lead to bad data in graph.

**Mitigation Strategy:**
- ✅ Pre-flight validation rejects corrupted files
- ✅ Partial extraction support with completeness scoring
- ✅ Post-load integrity audits catch violations
- ✅ Dead Letter Queue with detailed error metadata

**Monitoring:** Integrity audit pass rate (target: >95%), completeness score (target: >0.85)

---

### **Risk 5: Migration Complexity**
**Probability:** LOW | **Impact:** HIGH

**Scenario:** v1→v2 migration fails, data loss, downtime.

**Mitigation Strategy:**
- ✅ Explicit migration script with dry-run mode
- ✅ Full backup before migration
- ✅ Rollback script for quick restoration
- ✅ Test on copy of production data first

**Monitoring:** Migration success rate, data integrity checks post-migration

---

## 📈 Success Metrics

### **Quality Metrics**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Extraction Success Rate | >90% | (successful_extractions / total_attempts) |
| Average Confidence Score | >7.0 | AVG(confidence_score) across all fields |
| Average Completeness Score | >0.85 | AVG(completeness_score) across all documents |
| Integrity Audit Pass Rate | >95% | (audits_passed / total_audits) |
| Retry Success Rate | >50% | (successful_retries / total_retries) |

### **Performance Metrics**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Extraction Time (per PDF) | <3 sec | Time from PDF load to JSON save |
| Full Pipeline (3,500 PDFs) | <3 hours | End-to-end validation→extraction→loading |
| Query Response Time | <2 sec | Time from question to answer |
| Embedding Generation | <1 min/100 nodes | Batch embedding time |

### **Cost Metrics**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Average Cost per Extraction | <$0.05 | (total_llm_cost / successful_extractions) |
| Failed Extraction Rate | <5% | (failed_extractions / total_attempts) |
| Model Degradation Usage | <10% | (mini_model_uses / total_extractions) |

---

## 🚀 Implementation Checklist

### **Phase 1: Foundation** ✅
- [ ] Update requirements.txt (remove duckdb, openpyxl)
- [ ] Create directory structure (rejected_files/, failed_*, output/)
- [ ] Delete src/ingestion/structured_data.py
- [ ] Update .env.example with new config
- [ ] Create prompt template: prompts/v1.0_paf_extraction.txt

### **Phase 2: Schema & Models** ✅
- [ ] Update src/database/schema.py (new constraints, indexes, vector indexes)
- [ ] Create src/models/paf_document.py (Pydantic models)
- [ ] Create src/models/validation.py (ValidationResult, IntegrityViolation)
- [ ] Test schema initialization on clean database

### **Phase 3: Validation & Extraction** ✅
- [ ] Create src/validation/document_validator.py
- [ ] Create src/extraction/paf_extractor.py (with retry logic)
- [ ] Create src/extraction/prompt_manager.py
- [ ] Create src/testing/golden_test_generator.py
- [ ] Write unit tests for extraction pipeline
- [ ] Test with sample PAF documents

### **Phase 4: Loading & Integrity** ✅
- [ ] Create src/loading/graph_loader.py
- [ ] Create src/loading/integrity_audits.py
- [ ] Create src/loading/ingestion_tracker.py
- [ ] Update src/rag/embeddings.py (scope/cost embeddings)
- [ ] Write unit tests for loading pipeline
- [ ] Test with extracted JSON files

### **Phase 5: Query Engine** ✅
- [ ] Create src/query/query_classifier.py
- [ ] Refactor src/rag/query_interface.py → src/query_engine.py
- [ ] Implement multi-modal search (financial, semantic, temporal)
- [ ] Write unit tests for query engine
- [ ] Test with example queries

### **Phase 6: Integration & Testing** ✅
- [ ] Update main.py (remove structured data, add 3-phase pipeline)
- [ ] Write integration tests (end-to-end)
- [ ] Write performance tests (load testing)
- [ ] Create migration scripts (v1→v2, rollback)
- [ ] Update README.md
- [ ] Update QUICKSTART.md
- [ ] Create API.md (new)
- [ ] Final end-to-end testing with full dataset

---

## 📦 Final Deliverables

Upon completion of all 6 phases, the system will provide:

1. ✅ **Three-Phase PDF-Only Pipeline**
   - Validation → Extraction → Loading with explicit error boundaries
   - Dead Letter Queues for all failure modes
   - Comprehensive error metadata and logging

2. ✅ **Enhanced Neo4j Schema**
   - 9 node types (Project, Person, Company, Document, ScopeSummary, CostSummary, Revision, Chunk, IngestionRun)
   - 12 relationship types including temporal chains
   - 3 vector indexes (chunks, scope, cost)
   - Quality tracking properties (confidence, completeness, prompt_version)

3. ✅ **Multi-Modal Query Engine**
   - Financial queries (Cypher property filters)
   - Semantic queries (vector search across 3 node types)
   - Temporal queries (date-based Cypher)
   - Hybrid queries (intelligent combination)

4. ✅ **Production-Grade Error Handling**
   - Pre-flight validation (reject invalid PDFs)
   - Retry logic with exponential backoff
   - Model degradation (GPT-4o → GPT-4o-mini)
   - Post-load integrity audits

5. ✅ **Comprehensive Test Suite**
   - Unit tests (validator, extractor, loader, query engine)
   - Integration tests (end-to-end pipeline)
   - Auto-generated regression tests (golden files)
   - Performance tests (load testing)

6. ✅ **Audit Trail System**
   - IngestionRun tracking (when, what, how many)
   - Prompt versioning (data provenance)
   - Extraction metadata (model, timestamp, confidence)
   - Integrity violation reporting

7. ✅ **Migration Tools**
   - v1→v2 migration script with dry-run mode
   - Full backup before migration
   - Rollback script for quick restoration
   - Migration report generation

8. ✅ **Updated Documentation**
   - README.md (architecture, schema, examples)
   - QUICKSTART.md (PDF-only setup)
   - API.md (query engine API)
   - This document (REFACTORING_PLAN_V2.md)

---

## 📞 Next Steps

1. **Review this plan** with stakeholders
2. **Set up development environment** (Neo4j, OpenAI API key)
3. **Create feature branch**: `git checkout -b feature/graphrag-v2-refactor`
4. **Begin Phase 1** (Foundation & Cleanup) - estimated 2-3 days
5. **Use ultrathink/sequential thinking** for each phase implementation
6. **Regular checkpoints** after each phase completion
7. **Deploy to staging** for full dataset testing
8. **Production migration** with backup and rollback plan

---

**Document Status:** APPROVED
**Implementation Start Date:** TBD
**Target Completion Date:** TBD (2-3 weeks estimated)

---

**Let's build a production-grade PDF-only GraphRAG system! 🚀**
