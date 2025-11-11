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
        """Create uniqueness constraints for all primary entities (v2 schema)."""
        constraints = [
            # Core entities
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT company_name_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",

            # Document-related nodes
            "CREATE CONSTRAINT scope_id_unique IF NOT EXISTS FOR (s:ScopeSummary) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT cost_id_unique IF NOT EXISTS FOR (cs:CostSummary) REQUIRE cs.id IS UNIQUE",
            "CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (r:Revision) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",

            # Tracking & audit nodes
            "CREATE CONSTRAINT ingestion_run_id_unique IF NOT EXISTS FOR (ir:IngestionRun) REQUIRE ir.run_id IS UNIQUE",
        ]

        logger.info("Creating v2 uniqueness constraints...")
        for constraint in constraints:
            try:
                self.connection.execute_write(constraint)
                logger.debug(f"Constraint created: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint creation warning (may already exist): {e}")

        logger.info("V2 uniqueness constraints created successfully")

    def create_indexes(self) -> None:
        """Create indexes for frequently queried properties (v2 schema)."""
        indexes = [
            # Project indexes
            "CREATE INDEX project_name_index IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX project_confidence_index IF NOT EXISTS FOR (p:Project) ON (p.confidence_score)",

            # Document indexes
            "CREATE INDEX document_type_paf_index IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX document_hash_index IF NOT EXISTS FOR (d:Document) ON (d.file_hash)",

            # Revision indexes (for temporal queries)
            "CREATE INDEX revision_date_index IF NOT EXISTS FOR (r:Revision) ON (r.date)",

            # Cost indexes (for financial queries)
            "CREATE INDEX cost_total_index IF NOT EXISTS FOR (cs:CostSummary) ON (cs.total_request)",
        ]

        logger.info("Creating v2 property indexes...")
        for index in indexes:
            try:
                self.connection.execute_write(index)
                logger.debug(f"Index created: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation warning (may already exist): {e}")

        logger.info("V2 property indexes created successfully")

    def create_vector_index(
        self,
        index_name: str,
        node_label: str,
        property_name: str = "embedding",
        dimension: int = 384
    ) -> None:
        """
        Create vector index for RAG similarity search.

        Args:
            index_name: Name of the vector index
            node_label: Node label (e.g., 'Chunk', 'ScopeSummary', 'CostSummary')
            property_name: Property containing embedding vector (default: 'embedding')
            dimension: Embedding vector dimension (default: 384 for all-MiniLM-L6-v2)
        """
        # Validate identifiers for security (defense-in-depth)
        validate_identifier(index_name)
        validate_identifier(node_label)
        validate_identifier(property_name)

        # Create vector index
        vector_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON n.{property_name}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        logger.info(f"Creating vector index '{index_name}' on {node_label}.{property_name} (dim={dimension})...")
        try:
            self.connection.execute_write(vector_index_query)
            logger.info(f"Vector index '{index_name}' created successfully")
        except Exception as e:
            logger.warning(f"Vector index creation warning (may already exist): {e}")

    def create_all_vector_indexes(self, dimension: int = 384) -> None:
        """
        Create all vector indexes for v2 schema (chunks, scope, cost).

        Args:
            dimension: Embedding vector dimension (default: 384)
        """
        logger.info("Creating all v2 vector indexes...")

        # Chunk embeddings (for fine-grained search)
        self.create_vector_index(
            index_name="chunk_embeddings",
            node_label="Chunk",
            property_name="embedding",
            dimension=dimension
        )

        # Scope embeddings (for document-level scope search)
        self.create_vector_index(
            index_name="scope_embeddings",
            node_label="ScopeSummary",
            property_name="embedding",
            dimension=dimension
        )

        # Cost embeddings (for document-level cost/financial search)
        self.create_vector_index(
            index_name="cost_embeddings",
            node_label="CostSummary",
            property_name="embedding",
            dimension=dimension
        )

        logger.info("All v2 vector indexes created successfully")

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
        Initialize complete v2 database schema.

        Creates:
        - 9 uniqueness constraints (Project, Document, Person, Company, ScopeSummary, CostSummary, Revision, Chunk, IngestionRun)
        - 6 property indexes (project name/confidence, document type/hash, revision date, cost total)
        - 3 vector indexes (chunk, scope, cost embeddings)
        - 1 full-text index (chunk text)

        Args:
            vector_dimension: Dimension for vector embeddings (default: 384)
        """
        logger.info("Initializing v2 database schema...")
        self.create_constraints()
        self.create_indexes()
        self.create_all_vector_indexes(dimension=vector_dimension)
        self.create_chunk_text_index()
        logger.info("V2 database schema initialization complete")

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
