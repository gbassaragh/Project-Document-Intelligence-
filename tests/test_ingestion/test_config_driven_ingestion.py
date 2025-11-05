"""
Comprehensive tests for configuration-driven structured data ingestion.
Tests Phase 3 Tier 1.1 improvements: explicit file mapping instead of heuristic discovery.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from src.config.settings import FileMapping, StructuredDataIngestionConfig, Settings
from src.ingestion.structured_data import StructuredDataIngestion
from src.database.connection import Neo4jConnection


class TestConfigurationDrivenIngestion:
    """Test suite for configuration-driven ingestion."""

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
    def custom_settings(self, tmp_path):
        """Create test settings with custom file mappings."""
        settings = Settings()
        settings.data.structured_data_dir = tmp_path

        # Configure custom file mappings
        settings.structured_data_ingestion = StructuredDataIngestionConfig(
            file_mappings={
                "test_projects.csv": FileMapping(
                    table_name="projects",
                    entity_type="Project",
                    required_columns=["id", "name"]
                ),
                "test_people.xlsx": FileMapping(
                    table_name="people",
                    entity_type="Person",
                    required_columns=["name", "role"]
                ),
                "test_teams.csv": FileMapping(
                    table_name="teams",
                    entity_type="Team",
                    required_columns=["team_name", "member_name"]
                ),
            }
        )

        return settings

    def test_file_mapping_configuration(self, custom_settings):
        """Test that file mappings are correctly configured."""
        mappings = custom_settings.structured_data_ingestion.file_mappings

        assert "test_projects.csv" in mappings
        assert mappings["test_projects.csv"].table_name == "projects"
        assert mappings["test_projects.csv"].entity_type == "Project"
        assert "id" in mappings["test_projects.csv"].required_columns

    def test_load_all_files_with_explicit_mapping(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test load_all_files uses explicit configuration mapping."""
        # Create test CSV file
        test_file = tmp_path / "test_projects.csv"
        test_file.write_text("id,name,status\n1,Alpha,Active\n2,Beta,Planned\n")

        # Create ingestion with custom settings
        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)

            # Override settings
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Load files
            ingestion.load_all_files()

            # Verify table was loaded with correct name from configuration
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] if isinstance(table, tuple) else table for table in tables]

            assert "projects" in table_names
            assert "test_projects" not in table_names  # Should use configured name

    def test_load_all_files_skips_unmapped_files(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test that unmapped files are ignored."""
        # Create unmapped file
        unmapped_file = tmp_path / "random_data.csv"
        unmapped_file.write_text("col1,col2\nval1,val2\n")

        # Create mapped file
        mapped_file = tmp_path / "test_projects.csv"
        mapped_file.write_text("id,name\n1,Alpha\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] if isinstance(table, tuple) else table for table in tables]

            # Only mapped file should be loaded
            assert "projects" in table_names
            assert "random_data" not in table_names

    def test_load_all_files_handles_missing_files(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test graceful handling when configured files don't exist."""
        # No files created, but mappings configured
        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Should not raise, just log warnings
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0  # No tables loaded

    def test_ingest_persons_uses_configuration(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test ingest_persons uses entity_type from configuration."""
        # Create test file
        test_file = tmp_path / "test_people.xlsx"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "role": ["Manager", "Engineer", "Analyst"]
        })
        df.to_excel(test_file, index=False)

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Load files first
            ingestion.load_all_files()

            # Ingest persons
            ingestion.ingest_persons()

            # Verify execute_batch was called with Person data
            assert mock_connection.execute_batch.called
            call_args = mock_connection.execute_batch.call_args
            query = call_args[0][0]
            data = call_args[0][1]

            assert "MERGE (p:Person" in query
            assert len(data) == 3
            assert data[0]["name"] == "Alice"

    def test_ingest_teams_uses_configuration(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test ingest_teams uses entity_type from configuration."""
        # Create test file
        test_file = tmp_path / "test_teams.csv"
        test_file.write_text(
            "team_name,member_name\n"
            "Alpha Team,Alice\n"
            "Alpha Team,Bob\n"
            "Beta Team,Charlie\n"
        )

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Load files first
            ingestion.load_all_files()

            # Ingest teams
            ingestion.ingest_teams()

            # Verify execute_batch was called with Team data
            assert mock_connection.execute_batch.called
            call_args = mock_connection.execute_batch.call_args
            query = call_args[0][0]
            data = call_args[0][1]

            assert "MERGE (t:Team" in query
            assert len(data) == 3

    def test_no_heuristic_table_matching(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test that heuristic table name matching is not used."""
        # Create file with misleading name
        misleading_file = tmp_path / "project_manager_report.csv"
        misleading_file.write_text("name,role\nAlice,Manager\n")

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Load files
            ingestion.load_all_files()

            # This file should NOT be loaded because it's not in configuration
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] if isinstance(table, tuple) else table for table in tables]

            assert "project_manager_report" not in table_names
            assert len(table_names) == 0

    def test_table_name_validation_in_configuration(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test that table names from configuration are validated."""
        # Create file
        test_file = tmp_path / "test_projects.csv"
        test_file.write_text("id,name\n1,Alpha\n")

        # Modify configuration with invalid table name
        custom_settings.structured_data_ingestion.file_mappings["test_projects.csv"].table_name = (
            "projects'; DROP TABLE users--"
        )

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Should raise ValueError due to invalid table name
            with pytest.raises(ValueError, match="Invalid table name format"):
                ingestion.load_all_files()

    def test_empty_file_mappings_handling(
        self, mock_connection, tmp_path
    ):
        """Test handling when no file mappings are configured."""
        settings = Settings()
        settings.data.structured_data_dir = tmp_path
        settings.structured_data_ingestion = StructuredDataIngestionConfig(
            file_mappings={}
        )

        with patch("src.ingestion.structured_data.get_settings", return_value=settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = settings
            ingestion.data_dir = tmp_path

            # Should log warning and return early
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            assert len(tables) == 0

    def test_partial_file_loading_success(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test that pipeline continues when some files fail to load."""
        # Create one valid file
        valid_file = tmp_path / "test_projects.csv"
        valid_file.write_text("id,name\n1,Alpha\n")

        # Create one corrupted file
        corrupted_file = tmp_path / "test_people.xlsx"
        corrupted_file.write_text("not valid excel data")

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Should not raise, should continue with valid file
            ingestion.load_all_files()

            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] if isinstance(table, tuple) else table for table in tables]

            # Valid file should be loaded
            assert "projects" in table_names

    def test_configuration_prevents_accidental_ingestion(
        self, mock_connection, custom_settings, tmp_path
    ):
        """Test that configuration prevents accidental ingestion of wrong files.

        This is the critical fix: files like 'project_manager_report.csv' should NOT
        be automatically ingested even if they contain 'manager' or 'project' in the name.
        """
        # Create several files with potentially matching names
        files_to_create = [
            "project_manager_report.csv",
            "employee_roster.xlsx",
            "team_assignments_backup.csv",
            "person_data_old.xlsx",
        ]

        for filename in files_to_create:
            file_path = tmp_path / filename
            if filename.endswith(".csv"):
                file_path.write_text("col1,col2\nval1,val2\n")
            else:
                # Create minimal Excel file
                df = pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
                df.to_excel(file_path, index=False)

        with patch("src.ingestion.structured_data.get_settings", return_value=custom_settings):
            ingestion = StructuredDataIngestion(mock_connection)
            ingestion.settings = custom_settings
            ingestion.data_dir = tmp_path

            # Load files
            ingestion.load_all_files()

            # None of these files should be loaded
            tables = ingestion.duckdb_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] if isinstance(table, tuple) else table for table in tables]

            assert len(table_names) == 0  # No tables loaded
            assert "project_manager_report" not in table_names
            assert "employee_roster" not in table_names
            assert "team_assignments_backup" not in table_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
