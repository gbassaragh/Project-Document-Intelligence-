"""
Vector embeddings generation and Neo4j vector index management.
Creates embeddings for text chunks and manages the vector index for similarity search.
"""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection, validate_identifier

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings and vector index for RAG."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Initialize embedding manager.

        Args:
            connection: Neo4j connection instance
        """
        self.connection = connection
        self.settings = get_settings()

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.settings.processing.embedding_model}")
        self.model = SentenceTransformer(self.settings.processing.embedding_model)
        logger.info("Embedding model loaded successfully")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()

    def get_chunks_without_embeddings(self) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks that don't have embeddings yet.

        Returns:
            List of chunks with their IDs and text
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NULL
        RETURN c.chunk_id AS chunk_id, c.text AS text
        """
        chunks = self.connection.execute_query(query)
        logger.info(f"Found {len(chunks)} chunks without embeddings")
        return chunks

    def embed_all_chunks(self, batch_size: Optional[int] = None) -> None:
        """
        Generate and store embeddings for all chunks without embeddings.

        Args:
            batch_size: Number of chunks to process in each batch
        """
        if batch_size is None:
            batch_size = self.settings.processing.batch_size

        # Get chunks without embeddings
        chunks = self.get_chunks_without_embeddings()

        if not chunks:
            logger.info("All chunks already have embeddings")
            return

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Process in batches
        total_chunks = len(chunks)
        for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding chunks"):
            batch = chunks[i : i + batch_size]

            # Extract texts
            texts = [chunk["text"] for chunk in batch]

            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)

            # Prepare data for Neo4j
            embedding_data = [
                {"chunk_id": chunk["chunk_id"], "embedding": embedding}
                for chunk, embedding in zip(batch, embeddings)
            ]

            # Update chunks with embeddings
            query = """
            UNWIND $batch AS item
            MATCH (c:Chunk {chunk_id: item.chunk_id})
            SET c.embedding = item.embedding
            """
            self.connection.execute_write(query, {"batch": embedding_data})

        logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")

    def create_vector_index(self, index_name: str = "chunk_embeddings") -> None:
        """
        Create or recreate the vector index in Neo4j.

        Args:
            index_name: Name for the vector index
        """
        dimension = self.settings.processing.vector_dimension

        logger.info(f"Creating vector index '{index_name}' with dimension {dimension}...")

        # Validate index name for security (defense-in-depth)
        validate_identifier(index_name)

        # Drop existing index if it exists
        drop_query = f"DROP INDEX {index_name} IF EXISTS"
        try:
            self.connection.execute_write(drop_query)
        except Exception as e:
            logger.debug(f"Index drop warning: {e}")

        # Create new vector index
        create_query = f"""
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

        self.connection.execute_write(create_query)
        logger.info(f"Vector index '{index_name}' created successfully")

    def verify_embeddings(self) -> Dict[str, int]:
        """
        Verify embedding generation status.

        Returns:
            Dictionary with embedding statistics
        """
        query = """
        MATCH (c:Chunk)
        RETURN
            count(c) AS total_chunks,
            count(c.embedding) AS chunks_with_embeddings,
            count(CASE WHEN c.embedding IS NULL THEN 1 END) AS chunks_without_embeddings
        """
        result = self.connection.execute_query(query)

        stats = result[0] if result else {}
        logger.info(f"Embedding statistics: {stats}")
        return stats

    def similarity_search(
        self, query_text: str, top_k: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using the vector index.

        Args:
            query_text: Query text to search for
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of similar chunks with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query_text)

        # Vector similarity search query
        query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node, score
        WHERE score >= $score_threshold
        MATCH (node)<-[:HAS_CHUNK]-(d:Document)
        RETURN
            node.chunk_id AS chunk_id,
            node.text AS text,
            d.file_name AS document_name,
            d.type AS document_type,
            score
        ORDER BY score DESC
        """

        results = self.connection.execute_query(
            query,
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        )

        logger.debug(f"Found {len(results)} similar chunks for query")
        return results

    def run_full_embedding_pipeline(self) -> None:
        """Execute the complete embedding generation and indexing pipeline."""
        logger.info("Starting embedding generation pipeline...")

        try:
            # Generate embeddings for all chunks
            self.embed_all_chunks()

            # Create vector index
            self.create_vector_index()

            # Verify completion
            stats = self.verify_embeddings()

            logger.info("Embedding pipeline completed successfully")
            logger.info(
                f"Total chunks: {stats.get('total_chunks', 0)}, "
                f"With embeddings: {stats.get('chunks_with_embeddings', 0)}"
            )

        except Exception as e:
            logger.error(f"Embedding pipeline failed: {e}", exc_info=True)
            raise
