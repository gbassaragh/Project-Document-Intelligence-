# GraphRAG System Improvement Implementation Plan

## Overview
Systematic implementation of improvements identified in comprehensive code analysis.

## Implementation Strategy: 3-Wave Approach

### Wave 1: Performance & Reliability (HIGH IMPACT)
**Goal**: 5-10x performance improvement + prevent production failures
**Est. Time**: 6 hours
**Risk**: Medium (touching core extraction logic)

#### 1.1 Async Entity Extraction
- **File**: `src/extraction/entity_extractor.py`
- **Changes**:
  - Add async LLM client initialization
  - Convert `extract_from_chunks` to async with semaphore-based concurrency
  - Implement concurrent batch processing (10 parallel requests)
- **Expected Impact**: 5-10x faster entity extraction
- **Validation**: Time comparison before/after on 100 chunks

#### 1.2 API Rate Limiting & Retry Logic
- **Files**: `src/extraction/entity_extractor.py`, `requirements.txt`
- **Changes**:
  - Add `tenacity` dependency
  - Implement exponential backoff retry decorator
  - Add rate limiting wrapper for OpenAI API calls
- **Expected Impact**: Prevent API failures, handle transient errors
- **Validation**: Simulate API failures and verify retries

### Wave 2: Resource Management & Error Handling (RELIABILITY)
**Goal**: Prevent memory leaks + improve error visibility
**Est. Time**: 4 hours
**Risk**: Low (mostly additions, minimal changes to logic)

#### 2.1 Context Manager for DuckDB
- **File**: `src/ingestion/structured_data.py`
- **Changes**:
  - Add `__enter__` and `__exit__` methods
  - Ensure `close()` always called via context manager
- **Expected Impact**: Prevent memory leaks from unclosed connections
- **Validation**: Memory profiling before/after

#### 2.2 Transaction Rollback Support
- **File**: `src/database/connection.py`
- **Changes**:
  - Add explicit transaction management
  - Implement rollback on batch write failures
- **Expected Impact**: Data consistency on failures
- **Validation**: Test partial batch failure scenarios

#### 2.3 Enhanced Error Handling
- **Files**: `src/extraction/entity_extractor.py`, `src/rag/embeddings.py`
- **Changes**:
  - Differentiate transient vs. critical errors
  - Propagate critical errors instead of silent failures
  - Add structured error logging
- **Expected Impact**: Better debugging and user awareness
- **Validation**: Trigger errors and verify appropriate responses

### Wave 3: Code Quality & Testing (FOUNDATION)
**Goal**: 70% test coverage + fix type/style issues
**Est. Time**: 10 hours
**Risk**: Low (tests and style fixes don't affect functionality)

#### 3.1 Type Hint Fixes
- **Files**: `src/ingestion/pdf_parser.py`, others as needed
- **Changes**:
  - Fix lowercase `any` → `Any`
  - Add missing type hints
  - Run mypy for validation
- **Expected Impact**: Better type safety
- **Validation**: `mypy src/ --strict`

#### 3.2 PEP 8 Compliance
- **Files**: All Python files
- **Changes**:
  - Configure flake8
  - Fix line length violations
  - Fix import ordering
- **Expected Impact**: Consistent code style
- **Validation**: `flake8 src/ --max-line-length=100`

#### 3.3 Unit Tests (Priority Modules)
- **Files**: Create `tests/` structure
- **Test Coverage**:
  - `test_database/test_connection.py` (connection, batching, retries)
  - `test_extraction/test_entity_extractor.py` (extraction, error handling)
  - `test_rag/test_embeddings.py` (embedding generation, vector search)
  - `test_ingestion/test_structured_data.py` (DuckDB processing)
- **Expected Impact**: Catch regressions, enable safe refactoring
- **Validation**: `pytest tests/ --cov=src --cov-report=html`

## Implementation Order

### Priority 1 (Immediate - Wave 1)
1. ✅ Add tenacity to requirements.txt
2. ✅ Implement async entity extraction
3. ✅ Add rate limiting with exponential backoff
4. ✅ Test on sample dataset

### Priority 2 (This Sprint - Wave 2)
5. ✅ Add context manager to StructuredDataIngestion
6. ✅ Implement transaction rollback
7. ✅ Enhance error handling and propagation
8. ✅ Add structured logging

### Priority 3 (Next Sprint - Wave 3)
9. ✅ Fix all type hints (mypy clean)
10. ✅ Configure and fix PEP 8 violations
11. ✅ Create unit tests (70% coverage)
12. ✅ Add integration tests

## Validation Checklist

After each wave:
- [ ] Run existing functionality tests (manual)
- [ ] Check logs for new error patterns
- [ ] Run performance benchmarks
- [ ] Verify no regressions

After all waves:
- [ ] Run full pipeline on sample data
- [ ] Performance comparison (before/after)
- [ ] Memory profiling
- [ ] Security scan
- [ ] Documentation updates

## Rollback Strategy

Each wave is independent and can be rolled back:
- **Wave 1**: Revert async changes, keep sync version
- **Wave 2**: Revert context managers, keep explicit close()
- **Wave 3**: Tests don't affect production code

## Success Metrics

- ✅ **Performance**: 5x+ faster entity extraction
- ✅ **Reliability**: 0 API timeout failures over 1000 chunks
- ✅ **Resource Usage**: No memory leaks over 24hr run
- ✅ **Code Quality**: 70%+ test coverage, mypy clean
- ✅ **Maintainability**: PEP 8 compliant, flake8 clean

## Risk Mitigation

- Keep original methods as fallbacks during Wave 1
- Extensive logging during rollout
- Incremental deployment with monitoring
- Quick rollback plan for each wave
