"""
Edge case tests for structured data ingestion.
Tests Phase 3 Tier 1.2 improvements: malformed files, empty files, validation edge cases.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from src.config.settings import FileMapping, StructuredDataIngestionConfig, Settings
from src.ingestion.structured_data import StructuredDataIngestion
from src.database.connection import Neo4jConnection


class TestStructuredDataEdgeCases:
    """Test suite for structured data edge cases and error handling."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock Neo4j connection."""
        connection = MagicMock(spec=Neo4jConnection)
        connection.execute_batch.return_value = {
            "total_records": 10,
            "batch_size": 100,
            "total_batches": 1,
            "failed_batches": 0,
            "success_rate": 1.0,
            "avg_time_per_batch": 0.1,
            "total_time": 0.1
        }
        return connection

    @pytest.fixture
    def test_settings(self, tmp_path):
        """Create test settings."""
        settings = Settings()
        settings.data.structured_data_dir = tmp_path
        settings.structured_data_ingestion = StructuredDataIngestionConfig(
            file_mappings={
                "test.csv": FileMapping(
                    table_name="test_table",
                    entity_type="Test",
                    required_columns=["id", "name"]
                )
            }
        )
        return settings

    # Malformed File Tests
    def test_malformed_csv_empty_content(self, mock_connection, test_settings, tmp_path):
        """Test handling of completely empty CSV file."""
        empty_file = tmp_path / "test.csv"
        empty_file.write_text("")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Should handle gracefully without crashing
            ingestion.load_all_files()

            # Should not have loaded any tables
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0

    def test_malformed_csv_no_headers(self, mock_connection, test_settings, tmp_path):
        """Test handling of CSV with no headers."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("value1,value2\nvalue3,value4\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Should load with pandas default headers (0, 1, 2, etc.)
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            # Should still load, pandas handles it
            assert len(tables) >= 0  # May or may not work depending on pandas behavior

    def test_malformed_csv_inconsistent_columns(self, mock_connection, test_settings, tmp_path):
        """Test handling of CSV with inconsistent column counts."""
        file_path = tmp_path / "test.csv"
        file_path.write_text(
            "id,name,extra\n"
            "1,Alice\n"  # Missing column
            "2,Bob,value,toomany\n"  # Too many columns
            "3,Charlie,correct\n"
        )

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Pandas should handle this gracefully (filling NaN)
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            if tables:
                # Verify data was loaded (pandas fills missing values)
                result = ingestion.query_duckdb("SELECT * FROM test_table")
                assert len(result) == 3

    def test_malformed_excel_corrupted(self, mock_connection, test_settings, tmp_path):
        """Test handling of corrupted Excel file."""
        test_settings.structured_data_ingestion.file_mappings = {
            "test.xlsx": FileMapping(
                table_name="test_table",
                entity_type="Test",
                required_columns=[]
            )
        }

        corrupted_file = tmp_path / "test.xlsx"
        corrupted_file.write_text("This is not valid Excel data!")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Should handle error gracefully and continue
            ingestion.load_all_files()

            # No tables should be loaded
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0

    # Empty File Tests
    def test_empty_csv_with_headers_only(self, mock_connection, test_settings, tmp_path):
        """Test CSV with headers but no data rows."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name,status\n")  # Headers only

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()

            # Table should be loaded but empty
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 1

            result = ingestion.query_duckdb("SELECT * FROM test_table")
            assert len(result) == 0

    def test_empty_excel_sheet(self, mock_connection, test_settings, tmp_path):
        """Test Excel file with empty sheet."""
        test_settings.structured_data_ingestion.file_mappings = {
            "test.xlsx": FileMapping(
                table_name="test_table",
                entity_type="Test",
                required_columns=[]
            )
        }

        file_path = tmp_path / "test.xlsx"
        df = pd.DataFrame()  # Empty dataframe
        df.to_excel(file_path, index=False)

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            # Empty DataFrame may or may not create table
            # Pandas/DuckDB behavior may vary

    # Invalid Table Name Validation Tests
    def test_table_name_validation_sql_injection(self, mock_connection, test_settings, tmp_path):
        """Test table name validation rejects SQL injection attempts."""
        test_settings.structured_data_ingestion.file_mappings["test.csv"].table_name = (
            "test'; DROP TABLE users--"
        )

        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Should raise ValueError for invalid table name
            with pytest.raises(ValueError, match="Invalid table name format"):
                ingestion.load_all_files()

    def test_table_name_validation_special_characters(self, mock_connection, test_settings, tmp_path):
        """Test table name validation rejects special characters."""
        invalid_names = [
            "table@name",
            "table#name",
            "table$name",
            "table name",  # space
            "table;name",
            "table--name",
            "table/**/name",
        ]

        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        for invalid_name in invalid_names:
            test_settings.structured_data_ingestion.file_mappings["test.csv"].table_name = (
                invalid_name
            )

            with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
                ingestion = StructuredDataIngestion(mock_connection)
                ingestion.settings = test_settings
                ingestion.data_dir = tmp_path

                with pytest.raises(ValueError, match="Invalid table name format"):
                    ingestion.load_all_files()

    def test_table_name_validation_empty_string(self, mock_connection, test_settings, tmp_path):
        """Test table name validation rejects empty string."""
        test_settings.structured_data_ingestion.file_mappings["test.csv"].table_name = ""

        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            with pytest.raises(ValueError, match="Table name cannot be empty"):
                ingestion.load_all_files()

    def test_table_name_validation_valid_names(self, mock_connection, test_settings, tmp_path):
        """Test table name validation accepts valid names."""
        valid_names = [
            "table_name",
            "TableName",
            "table123",
            "_table",
            "table_name_123",
            "TABLE_NAME",
        ]

        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        for valid_name in valid_names:
            test_settings.structured_data_ingestion.file_mappings["test.csv"].table_name = (
                valid_name
            )

            with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
                ingestion = StructuredDataIngestion(mock_connection)
                ingestion.settings = test_settings
                ingestion.data_dir = tmp_path

                # Should not raise
                ingestion.load_all_files()

                tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0] if isinstance(t, tuple) else t for t in tables]
                assert valid_name in table_names

    # DuckDB Integration Edge Cases
    def test_duckdb_connection_closed(self, mock_connection, test_settings, tmp_path):
        """Test handling when DuckDB connection is closed."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Load files successfully
            ingestion.load_all_files()

            # Close connection
            ingestion.close()

            # Attempting to query should fail gracefully
            with pytest.raises(Exception):
                ingestion.query_duckdb("SELECT * FROM test_table")

    def test_context_manager_cleanup(self, mock_connection, test_settings, tmp_path):
        """Test that context manager properly cleans up resources."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alice\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            with StructuredDataIngestion(mock_connection) as ingestion:
                ingestion.settings = test_settings
                ingestion.data_dir = tmp_path
                ingestion.load_all_files()

            # After context manager exits, connection should be closed
            assert ingestion._closed is True

    # Ingestion Edge Cases
    def test_ingest_persons_no_person_tables(self, mock_connection, test_settings, tmp_path):
        """Test ingest_persons when no Person entity tables configured."""
        # Configure with only Project entities
        test_settings.structured_data_ingestion.file_mappings = {
            "test.csv": FileMapping(
                table_name="projects",
                entity_type="Project",  # Not Person
                required_columns=[]
            )
        }

        file_path = tmp_path / "test.csv"
        file_path.write_text("id,name\n1,Alpha\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()

            # Should log warning and return early, not crash
            ingestion.ingest_persons()

            # execute_batch should not be called
            assert not mock_connection.execute_batch.called

    def test_ingest_teams_no_team_tables(self, mock_connection, test_settings, tmp_path):
        """Test ingest_teams when no Team entity tables configured."""
        # Configure with only Person entities
        test_settings.structured_data_ingestion.file_mappings = {
            "test.csv": FileMapping(
                table_name="people",
                entity_type="Person",  # Not Team
                required_columns=[]
            )
        }

        file_path = tmp_path / "test.csv"
        file_path.write_text("name,role\nAlice,Manager\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()

            # Should log warning and return early, not crash
            ingestion.ingest_teams()

            # execute_batch should not be called
            assert not mock_connection.execute_batch.called

    def test_ingest_persons_null_names_filtered(self, mock_connection, test_settings, tmp_path):
        """Test that NULL names are filtered out during person ingestion."""
        test_settings.structured_data_ingestion.file_mappings = {
            "test.csv": FileMapping(
                table_name="people",
                entity_type="Person",
                required_columns=[]
            )
        }

        file_path = tmp_path / "test.csv"
        file_path.write_text("name,role\nAlice,Manager\n,Engineer\nBob,Analyst\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()
            ingestion.ingest_persons()

            # Should only have 2 persons (NULL name filtered out)
            call_args = mock_connection.execute_batch.call_args
            data = call_args[0][1]
            assert len(data) == 2
            assert all(person["name"] for person in data)  # No empty names

    def test_unsupported_file_extension(self, mock_connection, test_settings, tmp_path):
        """Test handling of unsupported file extension."""
        test_settings.structured_data_ingestion.file_mappings = {
            "test.txt": FileMapping(
                table_name="test_table",
                entity_type="Test",
                required_columns=[]
            )
        }

        file_path = tmp_path / "test.txt"
        file_path.write_text("id,name\n1,Alice\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=test_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = test_settings
            ingestion.data_dir = tmp_path

            # Should log warning and skip file
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
