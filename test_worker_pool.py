"""
Simple integration test for bounded worker pool implementation.
Tests worker pool with mock data to verify functionality.
"""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.extraction.entity_extractor_async import AsyncEntityExtractor, ExtractionResult, ExtractedEntity
from src.database.connection import Neo4jConnection


async def test_worker_pool_basic():
    """Test worker pool processes chunks correctly."""
    print("Testing bounded worker pool implementation...")

    # Mock connection
    mock_connection = MagicMock(spec=Neo4jConnection)
    mock_connection.execute_query.return_value = [
        {"chunk_id": f"chunk-{i}", "text": f"Test text {i}"}
        for i in range(25)  # Test with 25 chunks
    ]

    # Mock LLM response
    mock_result = ExtractionResult(
        entities=[ExtractedEntity(entity_type="Person", entity_value="Test", context="test context")],
        relationships=[]
    )

    # Create extractor with custom worker count
    with patch("src.extraction.entity_extractor_async.ChatOpenAI"):
        extractor = AsyncEntityExtractor(mock_connection, max_concurrent=5, worker_count=3)

        # Mock the extraction method
        async def mock_extract(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate processing time
            return mock_result

        extractor._extract_with_retry = mock_extract

        # Run extraction
        results = await extractor.extract_from_chunks_async()

        # Verify results
        assert len(results) == 25, f"Expected 25 results, got {len(results)}"
        assert extractor.stats["total_processed"] == 25
        print(f"✅ Processed {len(results)} chunks successfully")
        print(f"✅ Stats: {extractor.stats}")
        print(f"✅ Worker count: {extractor.worker_count}")


async def test_worker_count_validation():
    """Test worker count validation."""
    print("\nTesting worker count validation...")

    mock_connection = MagicMock(spec=Neo4jConnection)

    with patch("src.extraction.entity_extractor_async.ChatOpenAI"):
        # Valid worker counts
        try:
            extractor1 = AsyncEntityExtractor(mock_connection, worker_count=1)
            assert extractor1.worker_count == 1
            print("✅ worker_count=1 accepted")

            extractor2 = AsyncEntityExtractor(mock_connection, worker_count=50)
            assert extractor2.worker_count == 50
            print("✅ worker_count=50 accepted")

            # Default should match max_concurrent
            extractor3 = AsyncEntityExtractor(mock_connection, max_concurrent=15)
            assert extractor3.worker_count == 15
            print("✅ Default worker_count matches max_concurrent")

        except ValueError as e:
            print(f"❌ Unexpected validation error: {e}")
            raise

        # Invalid worker counts
        try:
            AsyncEntityExtractor(mock_connection, worker_count=0)
            print("❌ worker_count=0 should raise ValueError")
            assert False
        except ValueError:
            print("✅ worker_count=0 rejected correctly")

        try:
            AsyncEntityExtractor(mock_connection, worker_count=51)
            print("❌ worker_count=51 should raise ValueError")
            assert False
        except ValueError:
            print("✅ worker_count=51 rejected correctly")


async def test_large_chunk_set():
    """Test worker pool with large chunk set (memory efficiency)."""
    print("\nTesting with large chunk set (1000 chunks)...")

    mock_connection = MagicMock(spec=Neo4jConnection)
    mock_connection.execute_query.return_value = [
        {"chunk_id": f"chunk-{i}", "text": f"Text {i}"}
        for i in range(1000)
    ]

    mock_result = ExtractionResult(entities=[], relationships=[])

    with patch("src.extraction.entity_extractor_async.ChatOpenAI"):
        extractor = AsyncEntityExtractor(mock_connection, worker_count=10)

        # Mock extraction with minimal delay
        async def mock_extract(*args, **kwargs):
            await asyncio.sleep(0.001)
            return mock_result

        extractor._extract_with_retry = mock_extract

        # Run extraction
        import time
        start_time = time.time()
        results = await extractor.extract_from_chunks_async()
        elapsed = time.time() - start_time

        assert len(results) == 1000
        assert extractor.stats["total_processed"] == 1000
        print(f"✅ Processed 1000 chunks in {elapsed:.2f}s")
        print(f"✅ Average: {elapsed/1000*1000:.2f}ms per chunk")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("BOUNDED WORKER POOL INTEGRATION TESTS")
    print("=" * 60)

    try:
        await test_worker_count_validation()
        await test_worker_pool_basic()
        await test_large_chunk_set()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
