"""
PAF document extractor with LLM-based extraction, retry logic, and model degradation.

Implements robust extraction pipeline:
- Primary model: GPT-4o with 3 retry attempts
- Fallback model: GPT-4o-mini on persistent failures
- Exponential backoff: 2s, 4s, 8s delays
- DLQ support for failed extractions
- JSON validation with Pydantic models
- Completeness scoring and quality tracking
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from src.models.paf_document import PAFDocument
from src.models.validation import ExtractionError
from src.extraction.prompt_manager import PromptManager
from src.ingestion.pdf_parser import PDFParser
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class PAFExtractor:
    """Extracts structured data from PAF PDFs using LLM with retry logic."""

    def __init__(self):
        """Initialize PAF extractor with models and configuration."""
        self.settings = get_settings()

        # Initialize LLM models
        self.llm_primary = ChatOpenAI(
            model=self.settings.openai.model,
            temperature=0,
            max_retries=0  # We handle retries manually
        )
        self.llm_fallback = ChatOpenAI(
            model=self.settings.openai.fallback_model,
            temperature=0,
            max_retries=0
        )

        # Initialize components
        self.prompt_manager = PromptManager()
        self.pdf_parser = PDFParser()

        # Retry configuration
        self.max_attempts = self.settings.extraction.retry_max_attempts
        self.backoff_base = self.settings.extraction.retry_backoff_base
        self.enable_degradation = self.settings.extraction.enable_model_degradation

        # Output directory
        self.output_dir = self.settings.data.output_dir / "extracted_pafs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DLQ directory
        self.dlq_dir = self.settings.data.failed_extractions_dir
        self.dlq_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PAFExtractor initialized: "
            f"primary={self.settings.openai.model}, "
            f"fallback={self.settings.openai.fallback_model}, "
            f"max_attempts={self.max_attempts}"
        )

    def extract_with_retry(self, file_path: Path) -> PAFDocument:
        """
        Extract PAF document with retry logic and model degradation.

        Retry Strategy:
        1. Attempt 1-3: GPT-4o (primary) with exponential backoff (2s, 4s, 8s)
        2. Attempt 4 (if enabled): GPT-4o-mini (fallback)
        3. If all fail: Save to DLQ and raise ExtractionError

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted PAFDocument with quality metadata

        Raises:
            ExtractionError: If extraction fails after all retry attempts
        """
        file_path = Path(file_path)
        logger.info(f"Starting extraction: {file_path.name}")

        # Load prompt template
        prompt_version = self.settings.extraction.prompt_version
        prompt_template_text = self.prompt_manager.load_prompt(prompt_version)

        # Parse PDF text
        try:
            pdf_text = self.pdf_parser.extract_text(file_path)
            if not pdf_text or len(pdf_text.strip()) < 100:
                raise ValueError(f"Insufficient text extracted from PDF ({len(pdf_text)} chars)")
        except Exception as e:
            error_msg = f"PDF parsing failed: {str(e)}"
            logger.error(error_msg)
            self._save_to_dlq(file_path, error_msg, 0, [])
            raise ExtractionError(
                file_path=str(file_path),
                file_hash=self._calculate_file_hash(file_path),
                error_type="PDFParseError",
                error_message=error_msg,
                attempts=0,
                models_used=[],
                prompt_version=prompt_version
            )

        # Calculate file hash for tracking
        file_hash = self._calculate_file_hash(file_path)

        # Primary model attempts (GPT-4o)
        models_used = []
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(
                    f"Extraction attempt {attempt}/{self.max_attempts} "
                    f"with {self.settings.openai.model}"
                )

                result = self._extract_once(
                    pdf_text=pdf_text,
                    prompt_template=prompt_template_text,
                    llm=self.llm_primary,
                    file_path=file_path,
                    file_hash=file_hash,
                    prompt_version=prompt_version
                )

                models_used.append(self.settings.openai.model)
                logger.info(
                    f"✅ Extraction succeeded on attempt {attempt}: {file_path.name} "
                    f"(avg_confidence={result.calculate_average_confidence():.1f}, "
                    f"completeness={result.completeness_score:.2f})"
                )

                # Save to output directory
                self._save_extraction(result, file_path)
                return result

            except Exception as e:
                last_error = e
                models_used.append(self.settings.openai.model)
                logger.warning(
                    f"Extraction attempt {attempt} failed: {str(e)[:100]}"
                )

                # Exponential backoff before next retry
                if attempt < self.max_attempts:
                    delay = self.backoff_base ** attempt  # 2s, 4s, 8s
                    logger.debug(f"Waiting {delay}s before retry...")
                    time.sleep(delay)

        # Fallback to GPT-4o-mini if enabled
        if self.enable_degradation:
            try:
                logger.info(
                    f"Attempting fallback model: {self.settings.openai.fallback_model}"
                )

                result = self._extract_once(
                    pdf_text=pdf_text,
                    prompt_template=prompt_template_text,
                    llm=self.llm_fallback,
                    file_path=file_path,
                    file_hash=file_hash,
                    prompt_version=prompt_version
                )

                models_used.append(self.settings.openai.fallback_model)
                logger.info(
                    f"✅ Fallback extraction succeeded: {file_path.name} "
                    f"(avg_confidence={result.calculate_average_confidence():.1f}, "
                    f"completeness={result.completeness_score:.2f})"
                )

                # Save to output directory
                self._save_extraction(result, file_path)
                return result

            except Exception as e:
                last_error = e
                models_used.append(self.settings.openai.fallback_model)
                logger.error(f"Fallback extraction failed: {str(e)[:100]}")

        # All attempts failed - save to DLQ
        error_message = f"All extraction attempts failed. Last error: {str(last_error)}"
        logger.error(f"❌ {error_message}")

        extraction_error = ExtractionError(
            file_path=str(file_path),
            file_hash=file_hash,
            error_type=type(last_error).__name__,
            error_message=str(last_error),
            attempts=len(models_used),
            models_used=models_used,
            prompt_version=prompt_version
        )

        # Save to DLQ
        dlq_path = extraction_error.save_to_dlq(self.dlq_dir)
        logger.info(f"Extraction error saved to DLQ: {dlq_path}")

        raise extraction_error

    def _extract_once(
        self,
        pdf_text: str,
        prompt_template: str,
        llm: ChatOpenAI,
        file_path: Path,
        file_hash: str,
        prompt_version: str
    ) -> PAFDocument:
        """
        Perform single extraction attempt with given LLM.

        Args:
            pdf_text: Extracted PDF text
            prompt_template: Prompt template text
            llm: LangChain LLM instance
            file_path: Source PDF path
            file_hash: SHA256 hash of PDF
            prompt_version: Prompt version used

        Returns:
            Validated PAFDocument

        Raises:
            Exception: On extraction or validation failure
        """
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=PAFDocument)

        # Format prompt with PDF text
        prompt = PromptTemplate(
            template=prompt_template + "\n\nPDF Text:\n{pdf_text}\n\n{format_instructions}",
            input_variables=["pdf_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt.format(pdf_text=pdf_text)

        # Call LLM
        response = llm.invoke(formatted_prompt)

        # Parse response content
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_text = self._extract_json_from_response(response_text)

        # Parse JSON to dict
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")

        # Add extraction metadata
        data["extraction_model"] = llm.model_name
        data["extraction_timestamp"] = datetime.now()
        data["file_hash"] = file_hash
        data["prompt_version"] = prompt_version

        # Validate with Pydantic model
        try:
            paf_doc = PAFDocument(**data)
        except ValidationError as e:
            raise ValueError(f"Pydantic validation failed: {str(e)}")

        # Calculate and update completeness score
        calculated_completeness = paf_doc.calculate_completeness_score()
        paf_doc.completeness_score = calculated_completeness

        # Validate quality thresholds
        avg_confidence = paf_doc.calculate_average_confidence()
        if avg_confidence < self.settings.quality.confidence_threshold:
            logger.warning(
                f"Low confidence score: {avg_confidence:.1f} < "
                f"{self.settings.quality.confidence_threshold}"
            )

        if paf_doc.completeness_score < self.settings.quality.completeness_threshold:
            logger.warning(
                f"Low completeness score: {paf_doc.completeness_score:.2f} < "
                f"{self.settings.quality.completeness_threshold}"
            )

        return paf_doc

    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON from LLM response (handles markdown code blocks).

        Args:
            response_text: Raw LLM response

        Returns:
            Clean JSON text
        """
        # Remove markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            return response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            return response_text[start:end].strip()
        else:
            return response_text.strip()

    def _save_extraction(self, paf_doc: PAFDocument, source_file: Path) -> Path:
        """
        Save extracted PAF document to JSON file.

        Args:
            paf_doc: Extracted PAF document
            source_file: Source PDF path

        Returns:
            Path to saved JSON file
        """
        # Generate output filename: {original_stem}_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{source_file.stem}_{timestamp}.json"
        output_path = self.output_dir / output_filename

        # Save as pretty-printed JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                paf_doc.model_dump(mode='json'),
                f,
                indent=2,
                default=str  # Handle datetime serialization
            )

        logger.info(f"Saved extraction to: {output_path}")
        return output_path

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file for duplicate detection.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_to_dlq(
        self,
        file_path: Path,
        error_message: str,
        attempts: int,
        models_used: list[str]
    ) -> None:
        """
        Save extraction error to Dead Letter Queue.

        Args:
            file_path: Source PDF path
            error_message: Error description
            attempts: Number of attempts made
            models_used: List of models attempted
        """
        file_hash = self._calculate_file_hash(file_path)

        extraction_error = ExtractionError(
            file_path=str(file_path),
            file_hash=file_hash,
            error_type="ExtractionError",
            error_message=error_message,
            attempts=attempts,
            models_used=models_used,
            prompt_version=self.settings.extraction.prompt_version
        )

        dlq_path = extraction_error.save_to_dlq(self.dlq_dir)
        logger.info(f"Extraction error saved to DLQ: {dlq_path}")

    def extract_batch(
        self,
        file_paths: list[Path],
        continue_on_error: bool = True
    ) -> Tuple[list[PAFDocument], list[ExtractionError]]:
        """
        Extract batch of PAF documents.

        Args:
            file_paths: List of PDF paths to extract
            continue_on_error: Whether to continue batch on individual failures

        Returns:
            Tuple of (successful_extractions, failed_extractions)
        """
        successes = []
        failures = []

        logger.info(f"Starting batch extraction: {len(file_paths)} files")

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {file_path.name}")

            try:
                result = self.extract_with_retry(file_path)
                successes.append(result)
            except ExtractionError as e:
                failures.append(e)
                if not continue_on_error:
                    logger.error(f"Batch extraction stopped due to error: {str(e)}")
                    break
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")
                if not continue_on_error:
                    break

        logger.info(
            f"Batch extraction complete: "
            f"{len(successes)} succeeded, {len(failures)} failed"
        )

        return successes, failures
