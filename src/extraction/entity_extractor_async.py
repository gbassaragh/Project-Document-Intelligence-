"""
Async entity extraction using LLM (GPT-4o via LangChain) with rate limiting.
High-performance version with concurrent processing and retry logic.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm.asyncio import tqdm_asyncio
import openai

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Extracted entity from text."""

    entity_type: str = Field(
        description="Type of entity (Person, Project, Procedure, etc.)"
    )
    entity_value: str = Field(description="The actual entity value/name")
    context: Optional[str] = Field(description="Surrounding context from the text")


class ExtractedRelationship(BaseModel):
    """Extracted relationship between entities."""

    source_entity: str = Field(description="Source entity in the relationship")
    relationship_type: str = Field(description="Type of relationship")
    target_entity: str = Field(description="Target entity in the relationship")


class ExtractionResult(BaseModel):
    """Complete extraction result from a text chunk."""

    entities: List[ExtractedEntity] = Field(description="List of extracted entities")
    relationships: List[ExtractedRelationship] = Field(
        description="List of extracted relationships"
    )


class AsyncEntityExtractor:
    """Async entity extractor with concurrent processing and rate limiting."""

    def __init__(
        self, connection: Neo4jConnection, max_concurrent: int = 10
    ) -> None:
        """
        Initialize async entity extractor.

        Args:
            connection: Neo4j connection instance
            max_concurrent: Maximum number of concurrent API requests
        """
        self.connection = connection
        self.settings = get_settings()
        self.max_concurrent = max_concurrent

        # Initialize async LLM (ChatOpenAI supports async methods in newer versions)
        self.async_llm = ChatOpenAI(
            model=self.settings.openai.model,
            api_key=self.settings.openai.api_key,
            temperature=0,  # Deterministic extraction
            max_retries=0,  # We handle retries with tenacity
        )

        # Setup extraction prompt
        self.extraction_prompt = self._create_extraction_prompt()

        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }

    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Create the entity extraction prompt template.

        Returns:
            ChatPromptTemplate for extraction
        """
        parser = PydanticOutputParser(pydantic_object=ExtractionResult)

        template = """You are an expert at extracting structured information from project documents.

Extract the following types of entities and their relationships from the text:

**Entity Types:**
- Person: Names of people (project managers, engineers, stakeholders)
- Project: Project names or IDs
- Procedure: Standards, procedures, or guidelines (e.g., "AACE-101", "ISO 9001")
- Deliverable: Project deliverables or outputs
- Risk: Mentioned risks or issues

**Relationship Types:**
- MANAGES: Person manages Project
- MENTIONS: Document mentions Procedure/Person/Project
- REFERENCES: Document references another entity
- HAS_RISK: Project/Document has identified Risk

**Instructions:**
1. Be precise - only extract entities that are explicitly mentioned
2. Use consistent naming (proper capitalization, full names)
3. For procedures, extract the full procedure ID/name
4. Extract relationships only when clearly stated or strongly implied
5. If uncertain, omit rather than guess

**Text to analyze:**
{text}

{format_instructions}

Return the extracted entities and relationships in the specified JSON format.
"""

        return ChatPromptTemplate.from_template(template).partial(
            format_instructions=parser.get_format_instructions()
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            )
        ),
        reraise=True,
    )
    async def _extract_with_retry(self, text: str, chunk_id: str) -> ExtractionResult:
        """
        Extract entities with exponential backoff retry logic.

        Args:
            text: Text content to extract from
            chunk_id: Chunk identifier for logging

        Returns:
            ExtractionResult with entities and relationships
        """
        async with self.semaphore:  # Rate limiting
            try:
                # Format prompt
                messages = self.extraction_prompt.format_messages(text=text)

                # Call async LLM
                response = await self.async_llm.ainvoke(messages)

                # Parse response
                parser = PydanticOutputParser(pydantic_object=ExtractionResult)
                result = parser.parse(response.content)

                self.stats["successful"] += 1
                return result

            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                # Transient errors - retry will handle
                self.stats["retries"] += 1
                logger.warning(
                    f"Transient error for chunk {chunk_id}: {e}. Retrying..."
                )
                raise
            except Exception as e:
                # Critical errors - don't retry
                self.stats["failed"] += 1
                logger.error(f"Critical extraction error for chunk {chunk_id}: {e}")
                # Return empty result for non-transient errors
                return ExtractionResult(entities=[], relationships=[])

    async def extract_from_chunks_async(
        self, chunk_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from chunks using async concurrent processing.

        Args:
            chunk_ids: Optional list of specific chunk IDs to process

        Returns:
            List of extraction results with chunk IDs
        """
        logger.info("Starting async entity extraction...")

        # Query chunks from Neo4j
        if chunk_ids:
            query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            RETURN c.chunk_id AS chunk_id, c.text AS text
            """
            chunks = self.connection.execute_query(query, {"chunk_ids": chunk_ids})
        else:
            query = """
            MATCH (c:Chunk)
            RETURN c.chunk_id AS chunk_id, c.text AS text
            """
            chunks = self.connection.execute_query(query)

        logger.info(
            f"Processing {len(chunks)} chunks with {self.max_concurrent} concurrent requests..."
        )

        # Reset statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }

        # Create extraction tasks
        async def extract_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            """Extract entities from a single chunk."""
            self.stats["total_processed"] += 1
            extraction = await self._extract_with_retry(
                chunk["text"], chunk["chunk_id"]
            )
            return {
                "chunk_id": chunk["chunk_id"],
                "entities": [e.dict() for e in extraction.entities],
                "relationships": [r.dict() for r in extraction.relationships],
            }

        # Execute all extractions concurrently with progress bar
        tasks = [extract_chunk(chunk) for chunk in chunks]
        results = await tqdm_asyncio.gather(
            *tasks, desc="Extracting entities (async)"
        )

        # Log statistics
        logger.info(
            f"Async extraction completed: "
            f"{self.stats['successful']} successful, "
            f"{self.stats['failed']} failed, "
            f"{self.stats['retries']} retries"
        )

        return results

    def ingest_entities(self, extraction_results: List[Dict[str, Any]]) -> None:
        """
        Ingest extracted entities into Neo4j.

        Args:
            extraction_results: List of extraction results from chunks
        """
        logger.info("Ingesting extracted entities into Neo4j...")

        # Collect all entities by type
        persons = []
        projects = []
        procedures = []
        deliverables = []
        risks = []

        for result in extraction_results:
            chunk_id = result["chunk_id"]
            for entity in result["entities"]:
                entity_data = {
                    "value": entity["entity_value"],
                    "chunk_id": chunk_id,
                    "context": entity.get("context", ""),
                }

                if entity["entity_type"] == "Person":
                    persons.append(entity_data)
                elif entity["entity_type"] == "Project":
                    projects.append(entity_data)
                elif entity["entity_type"] == "Procedure":
                    procedures.append(entity_data)
                elif entity["entity_type"] == "Deliverable":
                    deliverables.append(entity_data)
                elif entity["entity_type"] == "Risk":
                    risks.append(entity_data)

        # Ingest persons
        if persons:
            query = """
            UNWIND $batch AS entity
            MERGE (p:Person {name: entity.value})
            WITH p, entity
            MATCH (c:Chunk {chunk_id: entity.chunk_id})
            MERGE (c)-[:MENTIONS_PERSON]->(p)
            """
            self.connection.execute_batch(
                query, persons, batch_size=self.settings.processing.batch_size
            )
            logger.info(f"Ingested {len(persons)} person mentions")

        # Ingest procedures
        if procedures:
            query = """
            UNWIND $batch AS entity
            MERGE (pr:Procedure {id: entity.value})
            WITH pr, entity
            MATCH (c:Chunk {chunk_id: entity.chunk_id})
            MERGE (c)-[:MENTIONS_PROCEDURE]->(pr)
            """
            self.connection.execute_batch(
                query, procedures, batch_size=self.settings.processing.batch_size
            )
            logger.info(f"Ingested {len(procedures)} procedure mentions")

        # Ingest deliverables
        if deliverables:
            query = """
            UNWIND $batch AS entity
            MERGE (d:Deliverable {name: entity.value})
            WITH d, entity
            MATCH (c:Chunk {chunk_id: entity.chunk_id})
            MERGE (c)-[:MENTIONS_DELIVERABLE]->(d)
            """
            self.connection.execute_batch(
                query, deliverables, batch_size=self.settings.processing.batch_size
            )
            logger.info(f"Ingested {len(deliverables)} deliverable mentions")

        logger.info("Entity ingestion completed")

    def ingest_relationships(self, extraction_results: List[Dict[str, Any]]) -> None:
        """
        Ingest extracted relationships into Neo4j.

        Args:
            extraction_results: List of extraction results from chunks
        """
        logger.info("Ingesting extracted relationships into Neo4j...")

        all_relationships = []
        for result in extraction_results:
            for rel in result["relationships"]:
                all_relationships.append(rel)

        if not all_relationships:
            logger.info("No relationships to ingest")
            return

        # Group by relationship type for efficient processing
        manages_rels = [
            r for r in all_relationships if r["relationship_type"] == "MANAGES"
        ]

        # Ingest MANAGES relationships
        if manages_rels:
            query = """
            UNWIND $batch AS rel
            MERGE (p:Person {name: rel.source_entity})
            MERGE (pr:Project {name: rel.target_entity})
            MERGE (p)-[:MANAGES]->(pr)
            """
            self.connection.execute_batch(
                query, manages_rels, batch_size=self.settings.processing.batch_size
            )
            logger.info(f"Ingested {len(manages_rels)} MANAGES relationships")

        logger.info("Relationship ingestion completed")

    async def run_full_extraction_async(
        self, chunk_ids: Optional[List[str]] = None
    ) -> None:
        """
        Execute the complete async entity extraction pipeline.

        Args:
            chunk_ids: Optional list of specific chunk IDs to process
        """
        logger.info("Starting async entity extraction pipeline...")

        try:
            # Extract entities from chunks (async)
            extraction_results = await self.extract_from_chunks_async(chunk_ids)

            # Ingest entities (sync - Neo4j operations)
            self.ingest_entities(extraction_results)

            # Ingest relationships (sync - Neo4j operations)
            self.ingest_relationships(extraction_results)

            logger.info("Async entity extraction pipeline completed successfully")

        except Exception as e:
            logger.error(f"Async entity extraction pipeline failed: {e}")
            raise
