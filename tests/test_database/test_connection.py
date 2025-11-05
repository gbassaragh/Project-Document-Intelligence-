"""
Unit tests for database connection module.
Tests connection management, query execution, batching, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.database.connection import (
    Neo4jConnection,
    get_connection,
    close_connection,
    validate_identifier,
)


class TestNeo4jConnection:
    """Test suite for Neo4jConnection class."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        driver.verify_connectivity = Mock()
        return driver

    @pytest.fixture
    def connection(self, mock_driver):
        """Create a Neo4jConnection instance with mocked driver."""
        with patch("src.database.connection.GraphDatabase.driver", return_value=mock_driver):
            conn = Neo4jConnection()
            conn.connect()
            return conn

    def test_connection_initialization(self):
        """Test connection initialization with settings."""
        conn = Neo4jConnection()
        assert conn.uri is not None
        assert conn.username is not None
        assert conn.database is not None
        assert conn._driver is None  # Not connected yet

    def test_successful_connection(self, mock_driver):
        """Test successful database connection."""
        with patch("src.database.connection.GraphDatabase.driver", return_value=mock_driver):
            conn = Neo4jConnection()
            conn.connect()

            mock_driver.verify_connectivity.assert_called_once()
            assert conn._driver is not None

    def test_connection_auth_error(self):
        """Test connection failure with authentication error."""
        with patch("src.database.connection.GraphDatabase.driver") as mock_gd:
            mock_gd.return_value.verify_connectivity.side_effect = AuthError("Invalid credentials")

            conn = Neo4jConnection()
            with pytest.raises(AuthError):
                conn.connect()

    def test_connection_service_unavailable(self):
        """Test connection failure when service is unavailable."""
        with patch("src.database.connection.GraphDatabase.driver") as mock_gd:
            mock_gd.return_value.verify_connectivity.side_effect = ServiceUnavailable("Service down")

            conn = Neo4jConnection()
            with pytest.raises(ServiceUnavailable):
                conn.connect()

    def test_close_connection(self, connection):
        """Test closing database connection."""
        connection.close()
        connection._driver.close.assert_called_once()

    def test_driver_property_before_connection(self):
        """Test accessing driver property before connecting."""
        conn = Neo4jConnection()
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = conn.driver

    def test_execute_query(self, connection):
        """Test executing a query and returning results."""
        mock_session = MagicMock()
        mock_result = [{"name": "Project A", "id": "1"}, {"name": "Project B", "id": "2"}]
        mock_session.run.return_value = [
            {"name": "Project A", "id": "1"},
            {"name": "Project B", "id": "2"},
        ]

        connection._driver.session.return_value.__enter__.return_value = mock_session

        query = "MATCH (p:Project) RETURN p.name as name, p.id as id"
        results = connection.execute_query(query)

        assert len(results) == 2
        mock_session.run.assert_called_once()

    def test_execute_write(self, connection):
        """Test executing a write transaction."""
        mock_session = MagicMock()
        connection._driver.session.return_value.__enter__.return_value = mock_session

        query = "CREATE (p:Project {id: $id, name: $name})"
        parameters = {"id": "1", "name": "Test Project"}

        connection.execute_write(query, parameters)

        mock_session.execute_write.assert_called_once()

    def test_execute_batch(self, connection):
        """Test batch execution with multiple records."""
        mock_session = MagicMock()
        connection._driver.session.return_value.__enter__.return_value = mock_session

        query = "UNWIND $batch AS item CREATE (p:Project {id: item.id})"
        batch_data = [{"id": f"proj-{i}"} for i in range(250)]

        connection.execute_batch(query, batch_data, batch_size=100)

        # Should be called 3 times (100, 100, 50)
        assert mock_session.execute_write.call_count == 3

    def test_execute_batch_with_failure(self, connection):
        """Test batch execution with partial failures."""
        mock_session = MagicMock()

        # Simulate failure on second batch
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Batch 2 failed")

        mock_session.execute_write.side_effect = side_effect
        connection._driver.session.return_value.__enter__.return_value = mock_session

        query = "UNWIND $batch AS item CREATE (p:Project {id: item.id})"
        batch_data = [{"id": f"proj-{i}"} for i in range(250)]

        with pytest.raises(RuntimeError, match="failed batches"):
            connection.execute_batch(query, batch_data, batch_size=100)

        # Should attempt all 3 batches despite failure
        assert mock_session.execute_write.call_count == 3

    def test_verify_connection_success(self, connection):
        """Test connection verification when connected."""
        connection._driver.verify_connectivity = Mock()
        assert connection.verify_connection() is True
        connection._driver.verify_connectivity.assert_called_once()

    def test_verify_connection_failure(self, connection):
        """Test connection verification when connection fails."""
        connection._driver.verify_connectivity.side_effect = Exception("Connection lost")
        assert connection.verify_connection() is False


class TestValidateIdentifier:
    """Test suite for Neo4j identifier validation (security hardening)."""

    def test_validate_safe_identifiers(self):
        """Test validation accepts safe Neo4j identifiers."""
        # These should all pass without exception
        safe_identifiers = [
            "my_index",
            "index123",
            "chunk_vector_index",
            "Index-Name-With-Dashes",
            "UPPERCASE_INDEX",
            "a",  # Single character
            "index_123_abc",
        ]
        for identifier in safe_identifiers:
            validate_identifier(identifier)  # Should not raise

    def test_validate_rejects_empty_identifier(self):
        """Test validation rejects empty identifiers."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_identifier("")

    def test_validate_rejects_too_long_identifier(self):
        """Test validation rejects overly long identifiers."""
        long_name = "a" * 100  # 100 characters, exceeds 64 limit
        with pytest.raises(ValueError, match="too long"):
            validate_identifier(long_name)

    def test_validate_rejects_sql_injection_attempts(self):
        """Test validation blocks SQL-style injection patterns."""
        injection_attempts = [
            "index'; DROP DATABASE",
            "index' OR '1'='1",
            "index'; DELETE FROM users--",
            "index\"; DROP TABLE",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError, match="Invalid identifier format"):
                validate_identifier(attempt)

    def test_validate_rejects_cypher_injection_attempts(self):
        """Test validation blocks Cypher injection patterns."""
        injection_attempts = [
            "index; MATCH (n) DELETE n",
            "index; DETACH DELETE (n)",
            "foo; MERGE (n:Node)",
            "bar; CREATE (n)",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError, match="Invalid identifier format"):
                validate_identifier(attempt)

    def test_validate_rejects_command_injection_attempts(self):
        """Test validation blocks shell command injection."""
        injection_attempts = [
            "index && rm -rf /",
            "index || cat /etc/passwd",
            "index; $(whoami)",
            "index`ls -la`",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError, match="Invalid identifier format"):
                validate_identifier(attempt)

    def test_validate_rejects_special_characters(self):
        """Test validation rejects identifiers with special characters."""
        special_chars = [
            "index@name",
            "index#123",
            "index$var",
            "index%percent",
            "index*wildcard",
            "index.dotted",
            "index/slash",
            "index\\backslash",
            "index name",  # Space
            "index\tname",  # Tab
            "index\nname",  # Newline
        ]
        for identifier in special_chars:
            with pytest.raises(ValueError, match="Invalid identifier format"):
                validate_identifier(identifier)

    def test_validate_respects_custom_max_length(self):
        """Test validation respects custom max_length parameter."""
        identifier = "a" * 50
        validate_identifier(identifier, max_length=64)  # Should pass
        with pytest.raises(ValueError, match="too long"):
            validate_identifier(identifier, max_length=30)  # Should fail


class TestGlobalConnection:
    """Test suite for global connection functions."""

    def test_get_connection_creates_singleton(self):
        """Test that get_connection creates a singleton."""
        # Reset global connection
        close_connection()

        with patch("src.database.connection.Neo4jConnection.connect"):
            conn1 = get_connection()
            conn2 = get_connection()

            # Should return same instance
            assert conn1 is conn2

    def test_close_connection_cleanup(self):
        """Test that close_connection cleans up properly."""
        with patch("src.database.connection.Neo4jConnection.connect"):
            conn = get_connection()
            close_connection()

            # Should close and reset to None
            conn.close.assert_called_once()
