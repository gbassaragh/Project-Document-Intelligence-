"""
PDF document parser and text extraction.
Processes PDF documents (PAFs, SRFs, IFRs) and extracts text content for entity extraction.
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from pypdf import PdfReader
from tqdm import tqdm

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a parsed document."""

    document_id: str
    file_path: str
    file_name: str
    document_type: str
    num_pages: int
    file_size: int


@dataclass
class ParsedDocument:
    """Parsed document with metadata and content."""

    metadata: DocumentMetadata
    text_content: str
    pages: List[str]


class PDFParser:
    """Handles PDF document parsing and text extraction."""

    def __init__(self, connection: Optional[Neo4jConnection] = None) -> None:
        """
        Initialize PDF parser.

        Args:
            connection: Optional Neo4j connection for direct ingestion
        """
        self.connection = connection
        self.settings = get_settings()
        self.pdf_dir = self.settings.data.pdf_data_dir

    @staticmethod
    def generate_document_id(file_path: Path) -> str:
        """
        Generate a unique document ID based on file path.

        Args:
            file_path: Path to the document

        Returns:
            Unique document ID (hash of file path)
        """
        return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]

    @staticmethod
    def extract_document_type(file_name: str) -> str:
        """
        Extract document type from file name.

        Args:
            file_name: Name of the file

        Returns:
            Document type (PAF, SRF, IFR, or Unknown)
        """
        file_name_upper = file_name.upper()
        if "PAF" in file_name_upper:
            return "PAF"
        elif "SRF" in file_name_upper:
            return "SRF"
        elif "IFR" in file_name_upper:
            return "IFR"
        elif "DESIGN" in file_name_upper:
            return "Design"
        elif "SPEC" in file_name_upper:
            return "Specification"
        else:
            return "Unknown"

    def parse_pdf(self, file_path: Path) -> ParsedDocument:
        """
        Parse a single PDF file and extract text content.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with metadata and content
        """
        logger.debug(f"Parsing PDF: {file_path.name}")

        try:
            # Read PDF
            reader = PdfReader(str(file_path))
            num_pages = len(reader.pages)

            # Extract text from each page
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                pages.append(text)

            # Combine all pages
            full_text = "\n\n".join(pages)

            # Generate metadata
            metadata = DocumentMetadata(
                document_id=self.generate_document_id(file_path),
                file_path=str(file_path),
                file_name=file_path.name,
                document_type=self.extract_document_type(file_path.name),
                num_pages=num_pages,
                file_size=file_path.stat().st_size,
            )

            return ParsedDocument(
                metadata=metadata, text_content=full_text, pages=pages
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}", exc_info=True)
            raise

    def parse_all_pdfs(self) -> List[ParsedDocument]:
        """
        Parse all PDF files in the PDF directory.

        Returns:
            List of parsed documents
        """
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
            return []

        pdf_files = list(self.pdf_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to parse")

        parsed_documents = []
        for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
            try:
                parsed_doc = self.parse_pdf(pdf_file)
                parsed_documents.append(parsed_doc)
            except Exception as e:
                logger.error(f"Skipping {pdf_file.name} due to error: {e}", exc_info=True)
                continue

        logger.info(f"Successfully parsed {len(parsed_documents)} documents")
        return parsed_documents

    def ingest_document_metadata(self, documents: List[ParsedDocument]) -> None:
        """
        Ingest document metadata into Neo4j.

        Args:
            documents: List of parsed documents
        """
        if not self.connection:
            raise ValueError("Neo4j connection required for ingestion")

        logger.info(f"Ingesting metadata for {len(documents)} documents...")

        # Prepare data for batch insertion
        documents_data = [
            {
                "id": doc.metadata.document_id,
                "type": doc.metadata.document_type,
                "path": doc.metadata.file_path,
                "file_name": doc.metadata.file_name,
                "num_pages": doc.metadata.num_pages,
                "file_size": doc.metadata.file_size,
            }
            for doc in documents
        ]

        # Batch insert documents
        query = """
        UNWIND $batch AS doc
        MERGE (d:Document {id: doc.id})
        SET d.type = doc.type,
            d.path = doc.path,
            d.file_name = doc.file_name,
            d.num_pages = doc.num_pages,
            d.file_size = doc.file_size
        """

        self.connection.execute_batch(
            query, documents_data, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(documents)} document metadata records")

    def chunk_document(
        self,
        document: ParsedDocument,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Split document into overlapping chunks for embedding.

        Args:
            document: Parsed document
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks

        Returns:
            List of chunks with metadata
        """
        text = document.text_content
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > 0:
                    end = start + break_point + 1
                    chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": f"{document.metadata.document_id}_chunk_{chunk_id}",
                    "document_id": document.metadata.document_id,
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_id,
                    "start_char": start,
                    "end_char": end,
                }
            )

            chunk_id += 1
            start = end - overlap  # Overlap for context continuity

        logger.debug(
            f"Created {len(chunks)} chunks for document {document.metadata.file_name}"
        )
        return chunks

    def ingest_chunks(
        self, documents: List[ParsedDocument], chunk_size: int = 1000, overlap: int = 200
    ) -> None:
        """
        Chunk all documents and ingest into Neo4j.

        Args:
            documents: List of parsed documents
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        if not self.connection:
            raise ValueError("Neo4j connection required for ingestion")

        logger.info(f"Chunking {len(documents)} documents...")

        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self.chunk_document(doc, chunk_size, overlap)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks")

        # Batch insert chunks
        query = """
        UNWIND $batch AS chunk
        MATCH (d:Document {id: chunk.document_id})
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.text = chunk.text,
            c.chunk_index = chunk.chunk_index,
            c.start_char = chunk.start_char,
            c.end_char = chunk.end_char
        MERGE (d)-[:HAS_CHUNK]->(c)
        """

        self.connection.execute_batch(
            query, all_chunks, batch_size=self.settings.processing.batch_size
        )

        logger.info(f"Ingested {len(all_chunks)} chunks into Neo4j")

    def run_full_pipeline(
        self, chunk_size: int = 1000, overlap: int = 200
    ) -> List[ParsedDocument]:
        """
        Execute the complete PDF parsing and ingestion pipeline.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks

        Returns:
            List of parsed documents
        """
        logger.info("Starting PDF parsing and ingestion pipeline...")

        try:
            # Parse all PDFs
            documents = self.parse_all_pdfs()

            if self.connection:
                # Ingest metadata
                self.ingest_document_metadata(documents)

                # Ingest chunks
                self.ingest_chunks(documents, chunk_size, overlap)

            logger.info("PDF parsing and ingestion pipeline completed successfully")
            return documents

        except Exception as e:
            logger.error(f"PDF parsing pipeline failed: {e}", exc_info=True)
            raise
