#!/usr/bin/env python3
"""
GraphRAG Knowledge System - Main Application
Orchestrates the entire pipeline for ingesting, indexing, and querying project documents.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.database.connection import get_connection, close_connection
from src.database.schema import SchemaManager
from src.ingestion.structured_data import StructuredDataIngestion
from src.ingestion.pdf_parser import PDFParser
from src.extraction.entity_extractor import EntityExtractor
from src.rag.embeddings import EmbeddingManager
from src.rag.query_interface import GraphRAGQuery


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    # Create logs directory
    settings.logging.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.logging.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("GraphRAG Knowledge System Starting")
    logger.info("=" * 80)


def initialize_database() -> None:
    """Initialize Neo4j database schema."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing database schema...")

    connection = get_connection()
    schema_manager = SchemaManager(connection)

    # Initialize schema with vector index
    settings = get_settings()
    schema_manager.initialize_schema(
        vector_dimension=settings.processing.vector_dimension
    )

    logger.info("Database schema initialized successfully")


def ingest_structured_data() -> None:
    """Ingest structured data from Excel/CSV files."""
    logger = logging.getLogger(__name__)
    logger.info("Starting structured data ingestion...")

    connection = get_connection()
    ingestion = StructuredDataIngestion(connection)
    ingestion.run_full_ingestion()

    logger.info("Structured data ingestion completed")


def ingest_documents() -> None:
    """Ingest and parse PDF documents."""
    logger = logging.getLogger(__name__)
    logger.info("Starting document ingestion...")

    connection = get_connection()
    parser = PDFParser(connection)
    documents = parser.run_full_pipeline(chunk_size=1000, overlap=200)

    logger.info(f"Document ingestion completed: {len(documents)} documents processed")


def extract_entities() -> None:
    """Extract entities from document chunks."""
    logger = logging.getLogger(__name__)
    logger.info("Starting entity extraction...")

    connection = get_connection()
    extractor = EntityExtractor(connection)
    extractor.run_full_extraction()

    logger.info("Entity extraction completed")


def generate_embeddings() -> None:
    """Generate embeddings for all chunks."""
    logger = logging.getLogger(__name__)
    logger.info("Starting embedding generation...")

    connection = get_connection()
    embedding_manager = EmbeddingManager(connection)
    embedding_manager.run_full_embedding_pipeline()

    logger.info("Embedding generation completed")


def run_full_pipeline() -> None:
    """Execute the complete end-to-end pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING FULL INGESTION PIPELINE")
    logger.info("=" * 80)

    try:
        # Step 1: Initialize database
        initialize_database()

        # Step 2: Ingest structured data
        ingest_structured_data()

        # Step 3: Ingest and parse documents
        ingest_documents()

        # Step 4: Extract entities
        extract_entities()

        # Step 5: Generate embeddings
        generate_embeddings()

        logger.info("=" * 80)
        logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def run_example_queries() -> None:
    """Run example queries to demonstrate the system."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING EXAMPLE QUERIES")
    logger.info("=" * 80)

    connection = get_connection()
    query_interface = GraphRAGQuery(connection)

    # Example queries
    queries = [
        "Find all projects that reference 'AACE procedures', had an 'IFR' issued, and were managed by someone from David Smith's team. Summarize the key risks.",
        "What procedures and standards are most frequently mentioned in project documents?",
        "List all projects with their assigned managers and team members.",
        "What are the common risks mentioned in IFR documents?",
        "Which projects have the most documentation available?",
    ]

    for i, question in enumerate(queries, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"QUERY {i}: {question}")
        logger.info(f"{'=' * 80}")

        try:
            result = query_interface.hybrid_query(question, top_k=5)

            logger.info(f"\n**ANSWER:**\n{result['answer']}\n")
            logger.info(f"**CONFIDENCE:** {result['confidence']}")
            logger.info(f"**RETRIEVED CHUNKS:** {len(result['chunks'])}")

            if result["chunks"]:
                logger.info(f"\n**SOURCE DOCUMENTS:**")
                for chunk in result["chunks"][:3]:
                    logger.info(
                        f"  - {chunk['document_name']} ({chunk['document_type']}) - Score: {chunk['score']:.2f}"
                    )

            logger.info(f"\n**RELATED ENTITIES:**")
            context = result["context"]
            if context.get("procedures"):
                logger.info(f"  Procedures: {', '.join(context['procedures'][:5])}")
            if context.get("persons"):
                logger.info(f"  People: {', '.join(context['persons'][:5])}")
            if context.get("projects"):
                logger.info(f"  Projects: {', '.join(context['projects'][:5])}")

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE QUERIES COMPLETED")
    logger.info("=" * 80)


def interactive_query_mode() -> None:
    """Start interactive query mode for user questions."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("INTERACTIVE QUERY MODE")
    logger.info("Type 'exit' or 'quit' to stop")
    logger.info("=" * 80)

    connection = get_connection()
    query_interface = GraphRAGQuery(connection)

    while True:
        try:
            question = input("\n🔍 Enter your question: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                logger.info("Exiting interactive mode")
                break

            if not question:
                continue

            print("\n⏳ Processing your query...\n")

            result = query_interface.hybrid_query(question, top_k=5)

            print("=" * 80)
            print(f"📝 ANSWER:")
            print("=" * 80)
            print(result["answer"])
            print()

            if result["chunks"]:
                print("=" * 80)
                print(f"📚 SOURCES ({len(result['chunks'])} documents):")
                print("=" * 80)
                for i, chunk in enumerate(result["chunks"][:3], 1):
                    print(
                        f"{i}. {chunk['document_name']} ({chunk['document_type']}) - "
                        f"Relevance: {chunk['score']:.1%}"
                    )
                print()

        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"\n❌ Error: {e}\n")


def main() -> None:
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Print menu
        print("\n" + "=" * 80)
        print("GraphRAG Knowledge System")
        print("=" * 80)
        print("\nOptions:")
        print("1. Run full ingestion pipeline (initialize + ingest all data)")
        print("2. Initialize database schema only")
        print("3. Ingest structured data only")
        print("4. Ingest documents only")
        print("5. Extract entities only")
        print("6. Generate embeddings only")
        print("7. Run example queries")
        print("8. Interactive query mode")
        print("9. Exit")
        print("=" * 80)

        choice = input("\nEnter your choice (1-9): ").strip()

        if choice == "1":
            run_full_pipeline()
        elif choice == "2":
            initialize_database()
        elif choice == "3":
            ingest_structured_data()
        elif choice == "4":
            ingest_documents()
        elif choice == "5":
            extract_entities()
        elif choice == "6":
            generate_embeddings()
        elif choice == "7":
            run_example_queries()
        elif choice == "8":
            interactive_query_mode()
        elif choice == "9":
            logger.info("Exiting application")
            return
        else:
            logger.warning("Invalid choice")
            return

        logger.info("\n" + "=" * 80)
        logger.info("OPERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        close_connection()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
