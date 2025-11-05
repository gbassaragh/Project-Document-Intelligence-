"""
Entity extraction using LLM (GPT-4o via LangChain).
Extracts entities and relationships from document text for knowledge graph population.

This module provides both sync and async extraction methods.
Use async version for better performance (5-10x faster).
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Extracted entity from text."""

    entity_type: str = Field(description="Type of entity (Person, Project, Procedure, etc.)")
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
    relationships: List[ExtractedRelationship] = Field(description="List of extracted relationships")


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Initialize entity extractor.

        Args:
            connection: Neo4j connection instance
        """
        self.connection = connection
        self.settings = get_settings()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.openai.model,
            api_key=self.settings.openai.api_key,
            temperature=0,  # Deterministic extraction
        )

        # Setup extraction prompt
        self.extraction_prompt = self._create_extraction_prompt()

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

    def extract_from_text(self, text: str) -> ExtractionResult:
        """
        Extract entities and relationships from a text chunk.

        Args:
            text: Text content to extract from

        Returns:
            ExtractionResult with entities and relationships
        """
        try:
            # Format prompt
            messages = self.extraction_prompt.format_messages(text=text)

            # Call LLM
            response = self.llm.invoke(messages)

            # Parse response
            parser = PydanticOutputParser(pydantic_object=ExtractionResult)
            result = parser.parse(response.content)

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return empty result on failure
            return ExtractionResult(entities=[], relationships=[])

    def extract_from_chunks(
        self, chunk_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from chunks stored in Neo4j.

        Args:
            chunk_ids: Optional list of specific chunk IDs to process

        Returns:
            List of extraction results with chunk IDs
        """
        logger.info("Extracting entities from chunks...")

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

        logger.info(f"Processing {len(chunks)} chunks for entity extraction...")

        results = []
        for chunk in tqdm(chunks, desc="Extracting entities"):
            extraction = self.extract_from_text(chunk["text"])
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "entities": [e.dict() for e in extraction.entities],
                    "relationships": [r.dict() for r in extraction.relationships],
                }
            )

        logger.info(f"Completed entity extraction for {len(results)} chunks")
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

    def run_full_extraction(
        self, chunk_ids: Optional[List[str]] = None, use_async: bool = True
    ) -> None:
        """
        Execute the complete entity extraction pipeline.

        Args:
            chunk_ids: Optional list of specific chunk IDs to process
            use_async: Use async version for better performance (default: True)
        """
        if use_async:
            logger.info("Using async entity extraction (5-10x faster)...")
            try:
                # Import async extractor
                from src.extraction.entity_extractor_async import AsyncEntityExtractor

                # Create async extractor
                async_extractor = AsyncEntityExtractor(
                    self.connection, max_concurrent=10
                )

                # Run async extraction
                asyncio.run(async_extractor.run_full_extraction_async(chunk_ids))

                logger.info("Async entity extraction completed successfully")
                return

            except ImportError as e:
                logger.warning(
                    f"Async extraction not available ({e}). Falling back to sync version."
                )
            except Exception as e:
                logger.error(f"Async extraction failed: {e}. Falling back to sync version.")

        # Fallback to sync version
        logger.info("Starting entity extraction pipeline (sync mode)...")

        try:
            # Extract entities from chunks
            extraction_results = self.extract_from_chunks(chunk_ids)

            # Ingest entities
            self.ingest_entities(extraction_results)

            # Ingest relationships
            self.ingest_relationships(extraction_results)

            logger.info("Entity extraction pipeline completed successfully")

        except Exception as e:
            logger.error(f"Entity extraction pipeline failed: {e}")
            raise
