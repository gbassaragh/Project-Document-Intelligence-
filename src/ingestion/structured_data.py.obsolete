"""
Structured data ingestion pipeline using DuckDB.
Processes Excel and CSV files containing project data, team rosters, and manager assignments.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import duckdb
import pandas as pd

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection

logger = logging.getLogger(__name__)


class StructuredDataIngestion:
    """Handles ingestion of structured data from Excel/CSV files using DuckDB.

    Supports context manager protocol for automatic resource cleanup.

    Example:
        with StructuredDataIngestion(connection) as ingestion:
            ingestion.run_full_ingestion()
    """

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Initialize structured data ingestion pipeline.

        Args:
            connection: Neo4j connection instance
        """
        self.connection = connection
        self.settings = get_settings()
        self.data_dir = self.settings.data.structured_data_dir
        self.duckdb_conn = duckdb.connect(":memory:")  # In-memory database
        self._closed = False

    def __enter__(self) -> "StructuredDataIngestion":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and ensure resources are cleaned up."""
        self.close()
        return False  # Don't suppress exceptions

    def _validate_table_name(self, table_name: str) -> None:
        """
        Validate table name for safe usage in SQL queries.

        This function validates table names used in dynamic SQL queries as a
        defense-in-depth measure to prevent SQL injection.

        Args:
            table_name: Table name to validate

        Raises:
            ValueError: If table name contains invalid characters

        Example:
            >>> self._validate_table_name("projects")  # OK
            >>> self._validate_table_name("'; DROP TABLE")  # Raises ValueError
        """
        if not table_name:
            raise ValueError("Table name cannot be empty")

        # Allow only safe characters for SQL identifiers
        # Must start with letter or underscore, followed by alphanumeric or underscore
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(
                f"Invalid table name format: '{table_name}'. "
                f"Table names must start with a letter or underscore and contain "
                f"only alphanumeric characters and underscores."
            )

    def load_excel_to_duckdb(self, file_path: Path, table_name: str) -> None:
        """
        Load Excel file into DuckDB table.

        Args:
            file_path: Path to Excel file
            table_name: Name for the DuckDB table
        """
        logger.info(f"Loading {file_path.name} into DuckDB table '{table_name}'...")
        try:
            # Validate table name before registration
            self._validate_table_name(table_name)

            df = pd.read_excel(file_path)
            self.duckdb_conn.register(table_name, df)
            logger.info(
                f"Loaded {len(df)} rows from {file_path.name} into '{table_name}'"
            )
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}", exc_info=True)
            raise

    def load_csv_to_duckdb(self, file_path: Path, table_name: str) -> None:
        """
        Load CSV file into DuckDB table.

        Args:
            file_path: Path to CSV file
            table_name: Name for the DuckDB table
        """
        logger.info(f"Loading {file_path.name} into DuckDB table '{table_name}'...")
        try:
            # Validate table name before registration
            self._validate_table_name(table_name)

            df = pd.read_csv(file_path)
            self.duckdb_conn.register(table_name, df)
            logger.info(
                f"Loaded {len(df)} rows from {file_path.name} into '{table_name}'"
            )
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}", exc_info=True)
            raise

    def load_all_files(self) -> None:
        """
        Load configured Excel and CSV files from the structured data directory.

        Uses explicit file mappings from configuration instead of loading all files.
        This prevents accidental ingestion of wrong files and provides clear schema validation.
        """
        if not self.data_dir.exists():
            logger.warning(f"Structured data directory does not exist: {self.data_dir}")
            return

        # Get file mappings from configuration
        file_mappings = self.settings.structured_data_ingestion.file_mappings

        if not file_mappings:
            logger.warning("No file mappings configured in settings. Skipping ingestion.")
            return

        # Track which configured files were found
        files_found = []
        files_missing = []

        for filename, mapping in file_mappings.items():
            file_path = self.data_dir / filename

            if not file_path.exists():
                files_missing.append(filename)
                logger.debug(f"Configured file not found: {filename}")
                continue

            files_found.append(filename)

            try:
                # Validate table name before loading
                self._validate_table_name(mapping.table_name)

                # Load based on file extension
                if filename.endswith('.xlsx'):
                    self.load_excel_to_duckdb(file_path, mapping.table_name)
                elif filename.endswith('.csv'):
                    self.load_csv_to_duckdb(file_path, mapping.table_name)
                else:
                    logger.warning(f"Unsupported file type: {filename}")
                    continue

                logger.info(
                    f"Loaded {filename} → table '{mapping.table_name}' "
                    f"(entity type: {mapping.entity_type})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to load {filename}: {e}. Skipping file.",
                    exc_info=True
                )
                continue

        # Log summary
        logger.info(
            f"File loading completed: {len(files_found)}/{len(file_mappings)} files loaded"
        )
        if files_missing:
            logger.info(f"Files not found: {', '.join(files_missing)}")

    def query_duckdb(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query on DuckDB and return results as DataFrame.

        Args:
            query: SQL query string

        Returns:
            Query results as pandas DataFrame
        """
        return self.duckdb_conn.execute(query).fetchdf()

    def ingest_projects(self) -> None:
        """
        Ingest project data into Neo4j.
        Expects a 'projects' table with columns: id, name, status
        """
        logger.info("Ingesting projects into Neo4j...")

        # Check if projects table exists
        tables = self.duckdb_conn.execute("SHOW TABLES").fetchall()
        if not any("projects" in str(table).lower() for table in tables):
            logger.warning("No 'projects' table found in DuckDB")
            return

        # Query all projects
        projects_df = self.query_duckdb("""
            SELECT
                CAST(id AS VARCHAR) as id,
                name,
                COALESCE(status, 'Unknown') as status
            FROM projects
        """)

        if projects_df.empty:
            logger.warning("No project data to ingest")
            return

        # Convert to list of dictionaries
        projects_data = projects_df.to_dict("records")

        # Batch insert into Neo4j using MERGE
        query = """
        UNWIND $batch AS project
        MERGE (p:Project {id: project.id})
        SET p.name = project.name,
            p.status = project.status
        """

        self.connection.execute_batch(
            query, projects_data, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(projects_data)} projects into Neo4j")

    def ingest_persons(self) -> None:
        """
        Ingest person data into Neo4j from configured tables.

        Uses explicit configuration mapping to identify Person entity tables.
        No heuristic table name matching.
        """
        logger.info("Ingesting persons into Neo4j...")

        # Get file mappings from configuration
        file_mappings = self.settings.structured_data_ingestion.file_mappings

        # Find all tables configured as Person entities
        person_tables = []
        for filename, mapping in file_mappings.items():
            if mapping.entity_type == "Person":
                person_tables.append(mapping.table_name)

        if not person_tables:
            logger.warning("No Person entity tables configured in settings")
            return

        # Check which tables actually exist in DuckDB
        existing_tables = self.duckdb_conn.execute("SHOW TABLES").fetchall()
        existing_table_names = {
            table[0] if isinstance(table, tuple) else table
            for table in existing_tables
        }

        # Filter to only tables that exist
        available_person_tables = [
            table for table in person_tables
            if table in existing_table_names
        ]

        if not available_person_tables:
            logger.warning(
                f"Configured Person tables not found in DuckDB: {person_tables}"
            )
            return

        # Query all persons from available tables
        all_persons = []
        for table_name in available_person_tables:
            # Validate table name for safe SQL construction
            self._validate_table_name(table_name)

            try:
                persons_df = self.query_duckdb(f"""
                    SELECT DISTINCT
                        name,
                        COALESCE(role, 'Unknown') as role
                    FROM {table_name}
                    WHERE name IS NOT NULL
                """)
                all_persons.append(persons_df)
                logger.debug(
                    f"Queried {len(persons_df)} persons from table '{table_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to query persons from table '{table_name}': {e}",
                    exc_info=True
                )
                continue

        if not all_persons:
            logger.warning("No person data retrieved from configured tables")
            return

        # Combine all person data
        import pandas as pd
        persons_df = pd.concat(all_persons, ignore_index=True).drop_duplicates()

        if persons_df.empty:
            logger.warning("No person data to ingest")
            return

        persons_data = persons_df.to_dict("records")

        # Batch insert into Neo4j
        query = """
        UNWIND $batch AS person
        MERGE (p:Person {name: person.name})
        SET p.role = person.role
        """

        self.connection.execute_batch(
            query, persons_data, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(persons_data)} persons into Neo4j")

    def ingest_teams(self) -> None:
        """
        Ingest team data and team membership relationships from configured tables.

        Uses explicit configuration mapping to identify Team entity tables.
        No heuristic table name matching.
        """
        logger.info("Ingesting teams into Neo4j...")

        # Get file mappings from configuration
        file_mappings = self.settings.structured_data_ingestion.file_mappings

        # Find all tables configured as Team entities
        team_tables = []
        for filename, mapping in file_mappings.items():
            if mapping.entity_type == "Team":
                team_tables.append(mapping.table_name)

        if not team_tables:
            logger.warning("No Team entity tables configured in settings")
            return

        # Check which tables actually exist in DuckDB
        existing_tables = self.duckdb_conn.execute("SHOW TABLES").fetchall()
        existing_table_names = {
            table[0] if isinstance(table, tuple) else table
            for table in existing_tables
        }

        # Filter to only tables that exist
        available_team_tables = [
            table for table in team_tables
            if table in existing_table_names
        ]

        if not available_team_tables:
            logger.warning(
                f"Configured Team tables not found in DuckDB: {team_tables}"
            )
            return

        # Query all teams from available tables
        all_teams = []
        for table_name in available_team_tables:
            # Validate table name for safe SQL construction
            self._validate_table_name(table_name)

            try:
                teams_df = self.query_duckdb(f"""
                    SELECT DISTINCT
                        team_name,
                        member_name
                    FROM {table_name}
                    WHERE team_name IS NOT NULL AND member_name IS NOT NULL
                """)
                all_teams.append(teams_df)
                logger.debug(
                    f"Queried {len(teams_df)} team memberships from table '{table_name}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to query teams from table '{table_name}': {e}",
                    exc_info=True
                )
                continue

        if not all_teams:
            logger.warning("No team data retrieved from configured tables")
            return

        # Combine all team data
        import pandas as pd
        teams_df = pd.concat(all_teams, ignore_index=True).drop_duplicates()

        if teams_df.empty:
            logger.warning("No team data to ingest")
            return

        teams_data = teams_df.to_dict("records")

        # Create teams and BELONGS_TO relationships
        query = """
        UNWIND $batch AS team
        MERGE (t:Team {name: team.team_name})
        MERGE (p:Person {name: team.member_name})
        MERGE (p)-[:BELONGS_TO]->(t)
        """

        self.connection.execute_batch(
            query, teams_data, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(teams_data)} team memberships into Neo4j")

    def ingest_project_assignments(self) -> None:
        """
        Ingest project manager assignments.
        Expects data with columns: project_id, manager_name
        """
        logger.info("Ingesting project assignments into Neo4j...")

        # Try to find assignments in various possible tables
        query_attempts = [
            """
            SELECT DISTINCT
                CAST(project_id AS VARCHAR) as project_id,
                manager_name
            FROM managers
            WHERE project_id IS NOT NULL AND manager_name IS NOT NULL
            """,
            """
            SELECT DISTINCT
                CAST(id AS VARCHAR) as project_id,
                manager as manager_name
            FROM projects
            WHERE id IS NOT NULL AND manager IS NOT NULL
            """,
        ]

        assignments_df = pd.DataFrame()
        for query in query_attempts:
            try:
                assignments_df = self.query_duckdb(query)
                if not assignments_df.empty:
                    break
            except Exception:
                continue

        if assignments_df.empty:
            logger.warning("No project assignment data found")
            return

        assignments_data = assignments_df.to_dict("records")

        # Create MANAGES relationships
        query = """
        UNWIND $batch AS assignment
        MATCH (p:Project {id: assignment.project_id})
        MERGE (person:Person {name: assignment.manager_name})
        MERGE (person)-[:MANAGES]->(p)
        """

        self.connection.execute_batch(
            query, assignments_data, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(assignments_data)} project assignments into Neo4j")

    def run_full_ingestion(self) -> None:
        """Execute the complete structured data ingestion pipeline."""
        logger.info("Starting full structured data ingestion pipeline...")

        try:
            # Load all files into DuckDB
            self.load_all_files()

            # Ingest into Neo4j in dependency order
            self.ingest_projects()
            self.ingest_persons()
            self.ingest_teams()
            self.ingest_project_assignments()

            logger.info("Structured data ingestion pipeline completed successfully")

        except Exception as e:
            logger.error(f"Structured data ingestion failed: {e}", exc_info=True)
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if not self._closed and self.duckdb_conn:
            try:
                self.duckdb_conn.close()
                self._closed = True
                logger.debug("DuckDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")
