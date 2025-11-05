"""
Unit tests for async entity extractor.
Tests async processing, rate limiting, and retry logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import openai

from src.extraction.entity_extractor_async import (
    AsyncEntityExtractor,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


@pytest.mark.asyncio
class TestAsyncEntityExtractor:
    """Test suite for AsyncEntityExtractor class."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock Neo4j connection."""
        connection = MagicMock()
        connection.execute_query = Mock(return_value=[
            {"chunk_id": "chunk1", "text": "Test text 1"},
            {"chunk_id": "chunk2", "text": "Test text 2"},
        ])
        connection.execute_batch = Mock()
        return connection

    @pytest.fixture
    def extractor(self, mock_connection):
        """Create an AsyncEntityExtractor instance."""
        with patch("src.extraction.entity_extractor_async.AsyncChatOpenAI"):
            return AsyncEntityExtractor(mock_connection, max_concurrent=2)

    def test_extractor_initialization(self, mock_connection):
        """Test extractor initialization."""
        with patch("src.extraction.entity_extractor_async.AsyncChatOpenAI"):
            extractor = AsyncEntityExtractor(mock_connection, max_concurrent=5)

            assert extractor.connection is mock_connection
            assert extractor.max_concurrent == 5
            assert extractor.semaphore._value == 5
            assert extractor.stats["total_processed"] == 0

    async def test_extract_with_retry_success(self, extractor):
        """Test successful extraction with retry logic."""
        # Mock successful LLM response
        mock_response = MagicMock()
        mock_response.content = """{
            "entities": [
                {"entity_type": "Person", "entity_value": "John Doe", "context": "manager"}
            ],
            "relationships": []
        }"""

        extractor.async_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await extractor._extract_with_retry("Test text", "chunk1")

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1
        assert result.entities[0].entity_value == "John Doe"
        assert extractor.stats["successful"] == 1

    async def test_extract_with_retry_rate_limit(self, extractor):
        """Test retry behavior on rate limit error."""
        # Mock rate limit error followed by success
        mock_response = MagicMock()
        mock_response.content = '{"entities": [], "relationships": []}'

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.RateLimitError("Rate limit exceeded")
            return mock_response

        extractor.async_llm.ainvoke = AsyncMock(side_effect=side_effect)

        result = await extractor._extract_with_retry("Test text", "chunk1")

        assert isinstance(result, ExtractionResult)
        assert call_count == 2  # Retried once
        assert extractor.stats["retries"] == 1

    async def test_extract_with_retry_max_attempts(self, extractor):
        """Test that extraction fails after max retry attempts."""
        # Mock persistent rate limit error
        extractor.async_llm.ainvoke = AsyncMock(
            side_effect=openai.RateLimitError("Rate limit exceeded")
        )

        with pytest.raises(openai.RateLimitError):
            await extractor._extract_with_retry("Test text", "chunk1")

        assert extractor.stats["retries"] >= 1

    async def test_extract_with_critical_error(self, extractor):
        """Test handling of non-retryable errors."""
        # Mock a critical error (not retryable)
        extractor.async_llm.ainvoke = AsyncMock(
            side_effect=ValueError("Invalid response format")
        )

        result = await extractor._extract_with_retry("Test text", "chunk1")

        # Should return empty result for critical errors
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0
        assert extractor.stats["failed"] == 1

    async def test_extract_from_chunks_concurrent(self, extractor, mock_connection):
        """Test concurrent extraction from multiple chunks."""
        # Mock successful extractions
        mock_response = MagicMock()
        mock_response.content = '{"entities": [], "relationships": []}'
        extractor.async_llm.ainvoke = AsyncMock(return_value=mock_response)

        results = await extractor.extract_from_chunks_async()

        assert len(results) == 2
        assert extractor.stats["total_processed"] == 2
        assert extractor.stats["successful"] == 2

    async def test_rate_limiting_with_semaphore(self, extractor):
        """Test that semaphore limits concurrent requests."""
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_extract(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return '{"entities": [], "relationships": []}'

        extractor.async_llm.ainvoke = AsyncMock(side_effect=mock_extract)

        # Create 10 chunks but limit to 2 concurrent
        extractor.connection.execute_query = Mock(return_value=[
            {"chunk_id": f"chunk{i}", "text": f"Text {i}"}
            for i in range(10)
        ])

        await extractor.extract_from_chunks_async()

        # Should never exceed max_concurrent (2)
        assert max_concurrent <= extractor.max_concurrent

    def test_ingest_entities(self, extractor, mock_connection):
        """Test entity ingestion into Neo4j."""
        extraction_results = [
            {
                "chunk_id": "chunk1",
                "entities": [
                    {"entity_type": "Person", "entity_value": "John Doe", "context": "manager"},
                    {"entity_type": "Procedure", "entity_value": "AACE-101", "context": ""},
                ],
                "relationships": [],
            }
        ]

        extractor.ingest_entities(extraction_results)

        # Should call execute_batch for persons and procedures
        assert mock_connection.execute_batch.call_count == 2

    def test_ingest_relationships(self, extractor, mock_connection):
        """Test relationship ingestion into Neo4j."""
        extraction_results = [
            {
                "chunk_id": "chunk1",
                "entities": [],
                "relationships": [
                    {
                        "source_entity": "John Doe",
                        "relationship_type": "MANAGES",
                        "target_entity": "Project Alpha",
                    }
                ],
            }
        ]

        extractor.ingest_relationships(extraction_results)

        # Should call execute_batch for relationships
        mock_connection.execute_batch.assert_called_once()

    def test_ingest_relationships_empty(self, extractor, mock_connection):
        """Test relationship ingestion with no relationships."""
        extraction_results = [
            {
                "chunk_id": "chunk1",
                "entities": [],
                "relationships": [],
            }
        ]

        extractor.ingest_relationships(extraction_results)

        # Should not call execute_batch
        mock_connection.execute_batch.assert_not_called()


class TestExtractionModels:
    """Test Pydantic models for extraction results."""

    def test_extracted_entity_creation(self):
        """Test ExtractedEntity model creation."""
        entity = ExtractedEntity(
            entity_type="Person",
            entity_value="John Doe",
            context="project manager",
        )

        assert entity.entity_type == "Person"
        assert entity.entity_value == "John Doe"
        assert entity.context == "project manager"

    def test_extracted_relationship_creation(self):
        """Test ExtractedRelationship model creation."""
        relationship = ExtractedRelationship(
            source_entity="John Doe",
            relationship_type="MANAGES",
            target_entity="Project Alpha",
        )

        assert relationship.source_entity == "John Doe"
        assert relationship.relationship_type == "MANAGES"
        assert relationship.target_entity == "Project Alpha"

    def test_extraction_result_creation(self):
        """Test ExtractionResult model creation."""
        entity = ExtractedEntity(
            entity_type="Person", entity_value="John Doe", context=None
        )
        relationship = ExtractedRelationship(
            source_entity="John Doe",
            relationship_type="MANAGES",
            target_entity="Project Alpha",
        )

        result = ExtractionResult(entities=[entity], relationships=[relationship])

        assert len(result.entities) == 1
        assert len(result.relationships) == 1
