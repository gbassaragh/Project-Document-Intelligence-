"""
Neo4j database connection management.
Provides connection pooling and session management for the GraphRAG system.
"""

import logging
import re
import threading
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
        """
        Establish connection to Neo4j database with optimized pool settings.

        Configures connection pool for production workloads:
        - Max pool size: 50 connections per host
        - Connection lifetime: 1 hour (prevents stale connections)
        - Liveness checks: 60s idle threshold
        - Connection timeout: 30s
        - Acquisition timeout: 60s
        """
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                # Connection pool optimization
                max_connection_pool_size=50,  # Limit per host (default: 100)
                max_connection_lifetime=3600,  # 1 hour (default: 3600)
                connection_timeout=30.0,  # 30s to establish TCP connection
                connection_acquisition_timeout=60.0,  # 60s to acquire from pool
                liveness_check_timeout=60.0,  # Test connections idle >60s
                keep_alive=True,  # Enable TCP keep-alive
                # Performance tuning
                fetch_size=1000,  # Records per batch (default: 1000)
            )
            # Verify connectivity
            self._driver.verify_connectivity()

            # Log connection pool configuration
            logger.info(
                f"Successfully connected to Neo4j at {self.uri} "
                f"(pool_size=50, lifetime=3600s, liveness_check=60s)"
            )
        except AuthError as e:
            logger.error(f"Authentication failed: {e}", exc_info=True)
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
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
                logger.error(f"Transaction failed and rolled back: {e}", exc_info=True)
                raise

    def calculate_optimal_batch_size(
        self, data_size: int, base_batch_size: int = 100
    ) -> int:
        """
        Calculate optimal batch size based on data characteristics.

        Uses adaptive sizing strategy:
        - Small datasets (<500): Use larger batches for efficiency
        - Medium datasets (500-5000): Use base batch size
        - Large datasets (>5000): Use smaller batches to prevent memory issues

        Args:
            data_size: Total number of records to process
            base_batch_size: Base batch size from configuration (default: 100)

        Returns:
            Optimized batch size (between 50 and 500)
        """
        if data_size < 500:
            # Small datasets: larger batches (up to 250)
            return min(data_size, 250)
        elif data_size < 5000:
            # Medium datasets: use base batch size
            return base_batch_size
        else:
            # Large datasets: smaller batches for memory efficiency
            return max(50, base_batch_size // 2)

    def execute_batch(
        self,
        query: str,
        batch_data: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        adaptive: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute batch writes using UNWIND for efficiency with transaction safety.

        Supports adaptive batch sizing that automatically adjusts batch size based on
        data characteristics and performance metrics.

        Args:
            query: Cypher query with UNWIND parameter
            batch_data: List of data dictionaries to process
            batch_size: Number of records per batch (None for adaptive sizing)
            adaptive: Enable adaptive batch sizing (default: True)

        Returns:
            Performance metrics dictionary with total_batches, failed_batches, avg_time_per_batch

        Raises:
            RuntimeError: If any batch fails (individual batch transactions are rolled back)
        """
        import time

        total = len(batch_data)

        # Determine batch size (adaptive or fixed)
        if batch_size is None or adaptive:
            batch_size = self.calculate_optimal_batch_size(total)
            logger.debug(f"Using adaptive batch size: {batch_size} for {total} records")
        else:
            logger.debug(f"Using fixed batch size: {batch_size} for {total} records")

        total_batches = (total + batch_size - 1) // batch_size
        failed_batches = []
        batch_times = []

        for i in range(0, total, batch_size):
            batch = batch_data[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                start_time = time.time()
                self.execute_write(query, {"batch": batch})
                elapsed = time.time() - start_time
                batch_times.append(elapsed)

                logger.debug(
                    f"Processed batch {batch_num}/{total_batches} "
                    f"({len(batch)} records in {elapsed:.3f}s)"
                )
            except Exception as e:
                logger.error(
                    f"Batch {batch_num}/{total_batches} failed: {e}. "
                    f"Transaction rolled back for this batch.",
                    exc_info=True
                )
                failed_batches.append(batch_num)
                # Continue processing remaining batches
                continue

        # Calculate performance metrics
        metrics = {
            "total_records": total,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "failed_batches": len(failed_batches),
            "success_rate": (total_batches - len(failed_batches)) / total_batches if total_batches > 0 else 0,
            "avg_time_per_batch": sum(batch_times) / len(batch_times) if batch_times else 0,
            "total_time": sum(batch_times),
        }

        logger.info(
            f"Batch processing completed: {total} records in {total_batches} batches "
            f"(success rate: {metrics['success_rate']:.1%}, avg time: {metrics['avg_time_per_batch']:.3f}s/batch)"
        )

        if failed_batches:
            raise RuntimeError(
                f"Batch processing completed with {len(failed_batches)} failed batches: "
                f"{failed_batches}. Failed batches were rolled back."
            )

        return metrics

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
            logger.error(f"Connection verification failed: {e}", exc_info=True)
            return False


# Global connection instance with thread-safe initialization
_connection: Optional[Neo4jConnection] = None
_connection_lock = threading.Lock()


def get_connection() -> Neo4jConnection:
    """
    Get or create the global Neo4j connection instance with thread-safe initialization.

    Uses double-checked locking pattern to ensure thread safety without
    excessive locking overhead.

    Returns:
        Neo4jConnection instance
    """
    global _connection
    if _connection is None:
        with _connection_lock:
            # Double-check locking: verify connection is still None inside the lock
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
