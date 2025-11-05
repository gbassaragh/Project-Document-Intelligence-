"""
Unit tests for adaptive batch sizing algorithm.
Tests Phase 2 calculate_optimal_batch_size() implementation.
"""

import pytest
from unittest.mock import MagicMock

from src.database.connection import Neo4jConnection


class TestAdaptiveBatchSizing:
    """Test suite for adaptive batch sizing algorithm."""

    @pytest.fixture
    def connection(self):
        """Create mock connection instance."""
        conn = Neo4jConnection()
        conn._driver = MagicMock()
        return conn

    # Test Small Datasets (<500 records)
    def test_small_dataset_single_record(self, connection):
        """Test batch size for single record."""
        batch_size = connection.calculate_optimal_batch_size(1)
        assert batch_size == 1
        assert batch_size <= 250

    def test_small_dataset_10_records(self, connection):
        """Test batch size for 10 records."""
        batch_size = connection.calculate_optimal_batch_size(10)
        assert batch_size == 10
        assert batch_size <= 250

    def test_small_dataset_100_records(self, connection):
        """Test batch size for 100 records."""
        batch_size = connection.calculate_optimal_batch_size(100)
        assert batch_size == 100
        assert batch_size <= 250

    def test_small_dataset_250_records(self, connection):
        """Test batch size for 250 records (max small)."""
        batch_size = connection.calculate_optimal_batch_size(250)
        assert batch_size == 250

    def test_small_dataset_499_records(self, connection):
        """Test batch size for 499 records (boundary)."""
        batch_size = connection.calculate_optimal_batch_size(499)
        assert batch_size == 250  # Capped at 250 for efficiency

    # Test Medium Datasets (500-5000 records)
    def test_medium_dataset_500_records(self, connection):
        """Test batch size for 500 records (boundary)."""
        batch_size = connection.calculate_optimal_batch_size(500)
        assert batch_size == 100  # Base batch size

    def test_medium_dataset_1000_records(self, connection):
        """Test batch size for 1000 records."""
        batch_size = connection.calculate_optimal_batch_size(1000)
        assert batch_size == 100

    def test_medium_dataset_2500_records(self, connection):
        """Test batch size for 2500 records (mid-range)."""
        batch_size = connection.calculate_optimal_batch_size(2500)
        assert batch_size == 100

    def test_medium_dataset_4999_records(self, connection):
        """Test batch size for 4999 records (boundary)."""
        batch_size = connection.calculate_optimal_batch_size(4999)
        assert batch_size == 100

    def test_medium_dataset_5000_records(self, connection):
        """Test batch size for 5000 records (boundary)."""
        batch_size = connection.calculate_optimal_batch_size(5000)
        assert batch_size == 100

    # Test Large Datasets (>5000 records)
    def test_large_dataset_5001_records(self, connection):
        """Test batch size for 5001 records (boundary)."""
        batch_size = connection.calculate_optimal_batch_size(5001)
        assert batch_size == 50  # base_batch_size // 2

    def test_large_dataset_10000_records(self, connection):
        """Test batch size for 10000 records."""
        batch_size = connection.calculate_optimal_batch_size(10000)
        assert batch_size == 50

    def test_large_dataset_100000_records(self, connection):
        """Test batch size for 100000 records."""
        batch_size = connection.calculate_optimal_batch_size(100000)
        assert batch_size == 50

    def test_large_dataset_million_records(self, connection):
        """Test batch size for 1 million records."""
        batch_size = connection.calculate_optimal_batch_size(1000000)
        assert batch_size == 50

    # Test Custom Base Batch Size
    def test_custom_base_batch_size_small(self, connection):
        """Test custom base batch size with small dataset."""
        batch_size = connection.calculate_optimal_batch_size(100, base_batch_size=50)
        assert batch_size == 100  # Still uses min(data_size, 250)

    def test_custom_base_batch_size_medium(self, connection):
        """Test custom base batch size with medium dataset."""
        batch_size = connection.calculate_optimal_batch_size(1000, base_batch_size=200)
        assert batch_size == 200  # Uses custom base

    def test_custom_base_batch_size_large(self, connection):
        """Test custom base batch size with large dataset."""
        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=200)
        assert batch_size == 100  # base_batch_size // 2

    def test_custom_base_batch_size_minimum_enforced(self, connection):
        """Test that minimum batch size of 50 is enforced for large datasets."""
        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=80)
        assert batch_size == 50  # max(50, 80 // 2) = max(50, 40) = 50

    # Test Edge Cases
    def test_zero_records(self, connection):
        """Test batch size for zero records."""
        batch_size = connection.calculate_optimal_batch_size(0)
        assert batch_size == 0

    def test_negative_records(self, connection):
        """Test batch size for negative records (invalid input)."""
        # Should still handle gracefully
        batch_size = connection.calculate_optimal_batch_size(-1)
        assert batch_size == -1  # Returns as-is (garbage in, garbage out)

    # Test Boundary Conditions
    def test_boundary_499_to_500(self, connection):
        """Test boundary between small and medium datasets."""
        batch_499 = connection.calculate_optimal_batch_size(499)
        batch_500 = connection.calculate_optimal_batch_size(500)

        assert batch_499 == 250  # Small dataset logic
        assert batch_500 == 100  # Medium dataset logic
        assert batch_499 > batch_500  # Transition happens

    def test_boundary_4999_to_5001(self, connection):
        """Test boundary between medium and large datasets."""
        batch_4999 = connection.calculate_optimal_batch_size(4999)
        batch_5000 = connection.calculate_optimal_batch_size(5000)
        batch_5001 = connection.calculate_optimal_batch_size(5001)

        assert batch_4999 == 100  # Medium dataset logic
        assert batch_5000 == 100  # Still medium (≤ 5000)
        assert batch_5001 == 50   # Large dataset logic
        assert batch_5000 > batch_5001  # Transition happens

    # Test Algorithm Properties
    def test_algorithm_monotonicity_for_small_datasets(self, connection):
        """Test that batch size increases with data size for small datasets."""
        sizes = [10, 50, 100, 200, 250]
        batch_sizes = [connection.calculate_optimal_batch_size(s) for s in sizes]

        # For small datasets, batch size should equal data size up to 250
        assert batch_sizes == sizes

    def test_algorithm_consistency(self, connection):
        """Test that algorithm returns consistent results."""
        data_size = 1000
        batch_size_1 = connection.calculate_optimal_batch_size(data_size)
        batch_size_2 = connection.calculate_optimal_batch_size(data_size)

        assert batch_size_1 == batch_size_2

    def test_algorithm_never_exceeds_250_for_small(self, connection):
        """Test that small dataset batch sizes never exceed 250."""
        for size in [100, 200, 300, 400, 499]:
            batch_size = connection.calculate_optimal_batch_size(size)
            assert batch_size <= 250

    def test_algorithm_respects_base_for_medium(self, connection):
        """Test that medium datasets use base_batch_size."""
        for size in [500, 1000, 2000, 3000, 5000]:
            batch_size = connection.calculate_optimal_batch_size(size, base_batch_size=100)
            assert batch_size == 100

    def test_algorithm_halves_base_for_large(self, connection):
        """Test that large datasets use half of base_batch_size (minimum 50)."""
        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=100)
        assert batch_size == 50  # 100 // 2

        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=200)
        assert batch_size == 100  # 200 // 2

        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=80)
        assert batch_size == 50  # max(50, 80 // 2) = 50

    # Test Performance Characteristics
    def test_small_dataset_optimization(self, connection):
        """Test that small datasets get larger batches for efficiency."""
        small_batch = connection.calculate_optimal_batch_size(250)
        medium_batch = connection.calculate_optimal_batch_size(500)

        # Small datasets should get larger relative batch sizes
        assert small_batch == 250
        assert medium_batch == 100
        assert small_batch > medium_batch

    def test_large_dataset_optimization(self, connection):
        """Test that large datasets get smaller batches for memory efficiency."""
        medium_batch = connection.calculate_optimal_batch_size(5000)
        large_batch = connection.calculate_optimal_batch_size(5001)

        # Large datasets should get smaller batches
        assert medium_batch == 100
        assert large_batch == 50
        assert medium_batch > large_batch

    # Test Documentation Examples
    def test_documentation_example_small(self, connection):
        """Test example from documentation: small dataset."""
        # "Small datasets (<500): Use larger batches for efficiency"
        batch_size = connection.calculate_optimal_batch_size(250)
        assert batch_size == 250

    def test_documentation_example_medium(self, connection):
        """Test example from documentation: medium dataset."""
        # "Medium datasets (500-5000): Use base batch size"
        batch_size = connection.calculate_optimal_batch_size(2500, base_batch_size=100)
        assert batch_size == 100

    def test_documentation_example_large(self, connection):
        """Test example from documentation: large dataset."""
        # "Large datasets (>5000): Use smaller batches to prevent memory issues"
        batch_size = connection.calculate_optimal_batch_size(10000, base_batch_size=100)
        assert batch_size == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
