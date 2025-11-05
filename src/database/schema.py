"""
Neo4j schema initialization and constraint management.
Creates all necessary constraints, indexes, and vector indexes for the GraphRAG system.
"""

import logging

from src.database.connection import Neo4jConnection, validate_identifier

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages Neo4j database schema, constraints, and indexes."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Initialize schema manager.

        Args:
            connection: Neo4j connection instance
        """
        self.connection = connection

    def create_constraints(self) -> None:
        """Create uniqueness constraints for all primary entities."""
        constraints = [
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT team_name_unique IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.id IS UNIQUE",
            "CREATE CONSTRAINT deliverable_name_unique IF NOT EXISTS FOR (d:Deliverable) REQUIRE d.name IS UNIQUE",
        ]

        logger.info("Creating uniqueness constraints...")
        for constraint in constraints:
            try:
                self.connection.execute_write(constraint)
                logger.debug(f"Constraint created: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint creation warning (may already exist): {e}")

        logger.info("Uniqueness constraints created successfully")

    def create_indexes(self) -> None:
        """Create indexes for frequently queried properties."""
        indexes = [
            "CREATE INDEX project_name_index IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX project_status_index IF NOT EXISTS FOR (p:Project) ON (p.status)",
            "CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX person_role_index IF NOT EXISTS FOR (p:Person) ON (p.role)",
        ]

        logger.info("Creating property indexes...")
        for index in indexes:
            try:
                self.connection.execute_write(index)
                logger.debug(f"Index created: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation warning (may already exist): {e}")

        logger.info("Property indexes created successfully")

    def create_vector_index(
        self, index_name: str = "chunk_embeddings", dimension: int = 384
    ) -> None:
        """
        Create vector index for RAG similarity search.

        Args:
            index_name: Name of the vector index
            dimension: Embedding vector dimension (default: 384 for all-MiniLM-L6-v2)
        """
        # Validate index name for security (defense-in-depth)
        validate_identifier(index_name)

        # Vector index for Chunk nodes with embeddings
        vector_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        logger.info(f"Creating vector index '{index_name}' with dimension {dimension}...")
        try:
            self.connection.execute_write(vector_index_query)
            logger.info(f"Vector index '{index_name}' created successfully")
        except Exception as e:
            logger.warning(f"Vector index creation warning (may already exist): {e}")

    def create_chunk_text_index(self) -> None:
        """Create full-text search index on Chunk text content."""
        fulltext_index_query = """
        CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """

        logger.info("Creating full-text search index on Chunk.text...")
        try:
            self.connection.execute_write(fulltext_index_query)
            logger.info("Full-text index created successfully")
        except Exception as e:
            logger.warning(f"Full-text index creation warning (may already exist): {e}")

    def drop_all_constraints(self) -> None:
        """Drop all constraints in the database. Use with caution!"""
        logger.warning("Dropping all constraints...")
        query = """
        SHOW CONSTRAINTS
        YIELD name
        RETURN name
        """
        constraints = self.connection.execute_query(query)

        for constraint in constraints:
            # Validate constraint name (from database, should be safe, but defense-in-depth)
            constraint_name = constraint['name']
            try:
                validate_identifier(constraint_name)
            except ValueError as e:
                logger.warning(f"Skipping invalid constraint name '{constraint_name}': {e}")
                continue

            drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
            self.connection.execute_write(drop_query)
            logger.debug(f"Dropped constraint: {constraint_name}")

        logger.info("All constraints dropped")

    def drop_all_indexes(self) -> None:
        """Drop all indexes in the database. Use with caution!"""
        logger.warning("Dropping all indexes...")
        query = """
        SHOW INDEXES
        YIELD name
        WHERE name IS NOT NULL
        RETURN name
        """
        indexes = self.connection.execute_query(query)

        for index in indexes:
            # Validate index name (from database, should be safe, but defense-in-depth)
            index_name = index['name']
            try:
                validate_identifier(index_name)
            except ValueError as e:
                logger.warning(f"Skipping invalid index name '{index_name}': {e}")
                continue

            drop_query = f"DROP INDEX {index_name} IF EXISTS"
            self.connection.execute_write(drop_query)
            logger.debug(f"Dropped index: {index_name}")

        logger.info("All indexes dropped")

    def clear_database(self) -> None:
        """Clear all nodes and relationships. Use with extreme caution!"""
        logger.warning("Clearing all data from database...")
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        self.connection.execute_write(query)
        logger.info("Database cleared")

    def initialize_schema(self, vector_dimension: int = 384) -> None:
        """
        Initialize complete database schema.

        Args:
            vector_dimension: Dimension for vector embeddings (default: 384)
        """
        logger.info("Initializing database schema...")
        self.create_constraints()
        self.create_indexes()
        self.create_vector_index(dimension=vector_dimension)
        self.create_chunk_text_index()
        logger.info("Database schema initialization complete")

    def get_schema_info(self) -> dict:
        """
        Get current schema information.

        Returns:
            Dictionary with constraints and indexes
        """
        constraints_query = "SHOW CONSTRAINTS YIELD name, type RETURN name, type"
        indexes_query = "SHOW INDEXES YIELD name, type, labelsOrTypes RETURN name, type, labelsOrTypes"

        constraints = self.connection.execute_query(constraints_query)
        indexes = self.connection.execute_query(indexes_query)

        return {"constraints": constraints, "indexes": indexes}
