"""
Neo4j database connection management.
Provides connection pooling and session management for the GraphRAG system.
"""

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def validate_identifier(identifier: str, max_length: int = 64) -> None:
    """
    Validate Neo4j identifier (index/constraint names) for safe usage in DDL.

    This function validates identifiers used in f-strings for administrative
    operations (CREATE INDEX, DROP CONSTRAINT, etc.) as a defense-in-depth measure.

    Note: All data queries use parameterized queries for actual injection protection.
    This validation is only for administrative identifiers that cannot be parameterized.

    Args:
        identifier: Index or constraint name to validate
        max_length: Maximum allowed length (default: 64)

    Raises:
        ValueError: If identifier contains invalid characters or is too long

    Example:
        >>> validate_identifier("my_index_123")  # OK
        >>> validate_identifier("index'; DROP DATABASE")  # Raises ValueError
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")

    if len(identifier) > max_length:
        raise ValueError(
            f"Identifier too long: {len(identifier)} characters (max: {max_length})"
        )

    # Allow only safe characters for Neo4j identifiers
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', identifier):
        raise ValueError(
            f"Invalid identifier format: '{identifier}'. "
            f"Only alphanumeric, underscore, and hyphen characters allowed."
        )


class Neo4jConnection:
    """Neo4j database connection manager with connection pooling."""

    def __init__(self) -> None:
        """Initialize Neo4j connection using settings from configuration."""
        settings = get_settings()
        self.uri = settings.neo4j.uri
        self.username = settings.neo4j.username
        self.password = settings.neo4j.password
        self.database = settings.neo4j.database
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver instance."""
        if not self._driver:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._driver

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        SECURITY: This method uses parameterized queries to prevent Cypher injection.
        Always pass user input as parameters, never embed it in the query string.

        Args:
            query: Cypher query string (use $param placeholders)
            parameters: Query parameters dictionary (values are safely escaped)

        Returns:
            List of result records as dictionaries

        Example:
            # SECURE: Using parameterized query
            conn.execute_query("MATCH (n) WHERE n.id = $id", {"id": user_input})

            # INSECURE: String interpolation (NEVER DO THIS!)
            # conn.execute_query(f"MATCH (n) WHERE n.id = '{user_input}'")
        """
        parameters = parameters or {}

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [dict(record) for record in result]

    def execute_write(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Execute a write transaction with automatic rollback on error.

        SECURITY: Uses parameterized queries for injection protection.

        Args:
            query: Cypher query string (use $param placeholders)
            parameters: Query parameters dictionary (values are safely escaped)

        Raises:
            Exception: If transaction fails (after automatic rollback)
        """
        parameters = parameters or {}

        with self.driver.session(database=self.database) as session:
            try:
                session.execute_write(lambda tx: tx.run(query, parameters))
            except Exception as e:
                logger.error(f"Transaction failed and rolled back: {e}")
                raise

    def execute_batch(
        self, query: str, batch_data: List[Dict[str, Any]], batch_size: int = 100
    ) -> None:
        """
        Execute batch writes using UNWIND for efficiency with transaction safety.

        Args:
            query: Cypher query with UNWIND parameter
            batch_data: List of data dictionaries to process
            batch_size: Number of records per batch

        Raises:
            Exception: If any batch fails (individual batch transactions are rolled back)
        """
        total = len(batch_data)
        total_batches = (total + batch_size - 1) // batch_size
        failed_batches = []

        for i in range(0, total, batch_size):
            batch = batch_data[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                self.execute_write(query, {"batch": batch})
                logger.debug(f"Processed batch {batch_num}/{total_batches}")
            except Exception as e:
                logger.error(
                    f"Batch {batch_num}/{total_batches} failed: {e}. "
                    f"Transaction rolled back for this batch."
                )
                failed_batches.append(batch_num)
                # Continue processing remaining batches
                continue

        if failed_batches:
            raise RuntimeError(
                f"Batch processing completed with {len(failed_batches)} failed batches: "
                f"{failed_batches}. Failed batches were rolled back."
            )

    def verify_connection(self) -> bool:
        """
        Verify that the database connection is active.

        Returns:
            True if connected, False otherwise
        """
        try:
            if self._driver:
                self._driver.verify_connectivity()
                return True
            return False
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False


# Global connection instance
_connection: Optional[Neo4jConnection] = None


def get_connection() -> Neo4jConnection:
    """
    Get or create the global Neo4j connection instance.

    Returns:
        Neo4jConnection instance
    """
    global _connection
    if _connection is None:
        _connection = Neo4jConnection()
        _connection.connect()
    return _connection


def close_connection() -> None:
    """Close the global Neo4j connection."""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
