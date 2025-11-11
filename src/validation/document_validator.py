"""
Document validator for pre-flight PDF validation (Phase 1: Validation).

Performs comprehensive validation checks before extraction:
- File existence and readability
- PDF validity (not corrupted)
- File name pattern matching
- File size bounds
- OCR/text layer presence
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import Optional
import shutil
import json
from datetime import datetime

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.models.validation import ValidationResult, ValidationStatus
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validates PDF documents before extraction."""

    # File size bounds (bytes)
    MIN_FILE_SIZE = 100 * 1024  # 100 KB
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    # PAF file name pattern (flexible - matches various PAF naming conventions)
    PAF_PATTERN = re.compile(r".*PAF.*\.pdf$", re.IGNORECASE)

    def __init__(self):
        """Initialize document validator."""
        self.settings = get_settings()
        self.rejected_dir = self.settings.data.rejected_files_dir

    def validate_pdf(self, file_path: Path) -> ValidationResult:
        """
        Perform comprehensive pre-flight validation on a PDF file.

        Validation checks:
        1. File exists and is readable
        2. File is a valid PDF (pypdf can open)
        3. File name matches PAF pattern
        4. File size within bounds (100KB < size < 50MB)
        5. PDF has text layer (OCR check)

        Args:
            file_path: Path to PDF file

        Returns:
            ValidationResult with status and reasons
        """
        reasons = []
        file_path = Path(file_path)

        logger.debug(f"Validating PDF: {file_path.name}")

        # Check 1: File exists and is readable
        if not file_path.exists():
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reasons=["File does not exist"],
                file_path=str(file_path),
                file_size=0,
                file_hash=""
            )

        if not file_path.is_file():
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                reasons=["Path is not a file"],
                file_path=str(file_path),
                file_size=0,
                file_hash=""
            )

        # Get file size and hash
        file_size = file_path.stat().st_size
        file_hash = self._calculate_file_hash(file_path)

        # Check 2: File size within bounds
        if file_size < self.MIN_FILE_SIZE:
            reasons.append(f"File too small ({file_size} bytes < {self.MIN_FILE_SIZE} bytes)")

        if file_size > self.MAX_FILE_SIZE:
            reasons.append(f"File too large ({file_size} bytes > {self.MAX_FILE_SIZE} bytes)")

        # Check 3: File name matches PAF pattern
        if not self.PAF_PATTERN.match(file_path.name):
            reasons.append(f"File name does not match PAF pattern (expected: *PAF*.pdf)")

        # Check 4: PDF validity
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                num_pages = len(reader.pages)

                if num_pages == 0:
                    reasons.append("PDF has zero pages")

                # Check 5: Text layer presence (OCR check)
                has_text = self._check_ocr_present(reader)
                if not has_text:
                    reasons.append("PDF has no extractable text (may be scanned image without OCR)")

        except PdfReadError as e:
            reasons.append(f"PDF is corrupted or invalid: {str(e)}")
        except Exception as e:
            reasons.append(f"Error reading PDF: {str(e)}")

        # Determine final status
        if reasons:
            # Check if any reasons are blocking vs. warnings
            blocking_keywords = ["corrupted", "invalid", "too large", "too small", "does not exist", "zero pages"]
            is_blocking = any(
                any(keyword in reason.lower() for keyword in blocking_keywords)
                for reason in reasons
            )

            status = ValidationStatus.REJECTED if is_blocking else ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        result = ValidationResult(
            status=status,
            reasons=reasons,
            file_path=str(file_path),
            file_size=file_size,
            file_hash=file_hash
        )

        if status == ValidationStatus.REJECTED:
            logger.warning(f"PDF validation failed: {file_path.name} - {'; '.join(reasons)}")
        elif status == ValidationStatus.WARNING:
            logger.info(f"PDF validation passed with warnings: {file_path.name} - {'; '.join(reasons)}")
        else:
            logger.debug(f"PDF validation passed: {file_path.name}")

        return result

    def _check_ocr_present(self, reader: PdfReader, sample_pages: int = 3) -> bool:
        """
        Check if PDF has extractable text (OCR present).

        Args:
            reader: PdfReader instance
            sample_pages: Number of pages to sample (default: 3)

        Returns:
            True if text is found, False otherwise
        """
        pages_to_check = min(sample_pages, len(reader.pages))

        for i in range(pages_to_check):
            try:
                text = reader.pages[i].extract_text()
                # Consider text present if we find at least 50 characters
                if text and len(text.strip()) > 50:
                    return True
            except Exception as e:
                logger.warning(f"Error extracting text from page {i}: {e}")
                continue

        return False

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
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def move_to_rejected(self, file_path: Path, validation_result: ValidationResult) -> Path:
        """
        Move rejected file to rejected_files directory with metadata.

        Creates a JSON metadata file alongside the rejected PDF with:
        - Rejection reasons
        - File hash
        - Validation timestamp
        - Original file path

        Args:
            file_path: Path to rejected file
            validation_result: Validation result with rejection reasons

        Returns:
            Path to rejected file in new location
        """
        file_path = Path(file_path)

        # Ensure rejected directory exists
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename if file already exists
        rejected_path = self.rejected_dir / file_path.name
        counter = 1
        while rejected_path.exists():
            stem = file_path.stem
            rejected_path = self.rejected_dir / f"{stem}_{counter}{file_path.suffix}"
            counter += 1

        # Move file
        try:
            shutil.move(str(file_path), str(rejected_path))
            logger.info(f"Moved rejected file: {file_path.name} -> {rejected_path}")
        except Exception as e:
            logger.error(f"Failed to move rejected file {file_path.name}: {e}")
            # If move fails, at least create metadata
            rejected_path = file_path

        # Create metadata file
        metadata_path = rejected_path.with_suffix('.json')
        metadata = {
            "original_path": str(file_path),
            "rejected_path": str(rejected_path),
            "rejection_reasons": validation_result.reasons,
            "file_size": validation_result.file_size,
            "file_hash": validation_result.file_hash,
            "validated_at": validation_result.validated_at.isoformat(),
            "status": validation_result.status.value
        }

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Created rejection metadata: {metadata_path.name}")
        except Exception as e:
            logger.error(f"Failed to create rejection metadata: {e}")

        return rejected_path

    def validate_batch(self, file_paths: list[Path]) -> tuple[list[Path], list[Path]]:
        """
        Validate a batch of PDF files.

        Args:
            file_paths: List of PDF file paths to validate

        Returns:
            Tuple of (valid_files, rejected_files)
        """
        valid_files = []
        rejected_files = []

        logger.info(f"Validating batch of {len(file_paths)} files...")

        for file_path in file_paths:
            result = self.validate_pdf(file_path)

            if result.is_valid():
                valid_files.append(file_path)
            else:
                # Move rejected file to DLQ
                rejected_path = self.move_to_rejected(file_path, result)
                rejected_files.append(rejected_path)

        logger.info(f"Batch validation complete: {len(valid_files)} valid, {len(rejected_files)} rejected")

        return valid_files, rejected_files
