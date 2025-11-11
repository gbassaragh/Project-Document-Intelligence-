"""
Validation models for PAF document processing pipeline.

These models define validation results, integrity violations, and error tracking
throughout the three-phase pipeline (Validation → Extraction → Loading).
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime
from pathlib import Path


class ValidationStatus(str, Enum):
    """Status of document validation."""

    VALID = "valid"
    REJECTED = "rejected"
    WARNING = "warning"


class ValidationResult(BaseModel):
    """Result of pre-flight document validation (Phase 1)."""

    status: ValidationStatus = Field(description="Validation outcome")
    reasons: List[str] = Field(default_factory=list, description="List of validation issues or warnings")
    file_path: str = Field(description="Path to validated file")
    file_size: int = Field(description="File size in bytes")
    file_hash: str = Field(description="SHA256 hash of file contents")
    validated_at: datetime = Field(default_factory=datetime.now, description="When validation was performed")

    def is_valid(self) -> bool:
        """Check if validation passed.

        Returns:
            True if status is VALID or WARNING
        """
        return self.status in [ValidationStatus.VALID, ValidationStatus.WARNING]

    def get_rejection_summary(self) -> str:
        """Get human-readable rejection summary.

        Returns:
            Summary string with all rejection reasons
        """
        if self.status == ValidationStatus.VALID:
            return "File passed all validation checks"
        elif self.status == ValidationStatus.WARNING:
            return f"File passed with warnings: {'; '.join(self.reasons)}"
        else:
            return f"File rejected: {'; '.join(self.reasons)}"


class IntegrityViolationType(str, Enum):
    """Types of integrity violations in Neo4j graph."""

    ORPHANED_PROJECT = "orphaned_project"  # Project without Scope
    MISSING_COST = "missing_cost"  # Project without CostSummary
    DUPLICATE_PROJECT = "duplicate_project"  # Multiple projects with same ID
    INVALID_TOTAL = "invalid_total"  # Cost total <= 0 or NULL
    LOW_CONFIDENCE = "low_confidence"  # Extraction confidence < threshold
    MISSING_DOCUMENT = "missing_document"  # Project without Document
    BROKEN_REVISION_CHAIN = "broken_revision_chain"  # Revision chain has gaps
    INVALID_EMBEDDING = "invalid_embedding"  # Embedding dimension mismatch


class ViolationSeverity(str, Enum):
    """Severity levels for integrity violations."""

    ERROR = "error"  # Critical issue requiring immediate attention
    WARNING = "warning"  # Non-critical issue that should be reviewed
    INFO = "info"  # Informational notice


class IntegrityViolation(BaseModel):
    """Represents an integrity violation found during post-load audits."""

    violation_type: IntegrityViolationType = Field(description="Type of integrity violation")
    entity_id: str = Field(description="ID of affected entity")
    description: str = Field(description="Human-readable description of the violation")
    severity: ViolationSeverity = Field(description="Severity level")
    detected_at: datetime = Field(default_factory=datetime.now, description="When violation was detected")
    metadata: Optional[dict] = Field(None, description="Additional context about the violation")

    def get_formatted_message(self) -> str:
        """Get formatted violation message for logging.

        Returns:
            Formatted message with severity, type, and details
        """
        severity_symbol = {
            ViolationSeverity.ERROR: "❌",
            ViolationSeverity.WARNING: "⚠️",
            ViolationSeverity.INFO: "ℹ️"
        }
        symbol = severity_symbol.get(self.severity, "")
        return f"{symbol} [{self.severity.value.upper()}] {self.violation_type.value}: {self.description} (entity: {self.entity_id})"


class AuditReport(BaseModel):
    """Report from post-load integrity audits."""

    audit_run_id: str = Field(description="Unique identifier for this audit run")
    violations: List[IntegrityViolation] = Field(default_factory=list, description="List of violations found")
    total_entities_checked: int = Field(description="Total number of entities audited")
    run_at: datetime = Field(default_factory=datetime.now, description="When audit was performed")

    def has_errors(self) -> bool:
        """Check if audit found any ERROR-level violations.

        Returns:
            True if any ERROR violations exist
        """
        return any(v.severity == ViolationSeverity.ERROR for v in self.violations)

    def error_count(self) -> int:
        """Count ERROR-level violations.

        Returns:
            Number of ERROR violations
        """
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.ERROR)

    def warning_count(self) -> int:
        """Count WARNING-level violations.

        Returns:
            Number of WARNING violations
        """
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.WARNING)

    def summary(self) -> str:
        """Get human-readable audit summary.

        Returns:
            Summary string with violation counts
        """
        errors = self.error_count()
        warnings = self.warning_count()
        return (
            f"Audit Report ({self.audit_run_id}):\n"
            f"  Entities checked: {self.total_entities_checked}\n"
            f"  Errors: {errors}\n"
            f"  Warnings: {warnings}\n"
            f"  Total violations: {len(self.violations)}"
        )


class ExtractionError(BaseModel):
    """Error metadata for failed extractions (DLQ)."""

    file_path: str = Field(description="Path to PDF that failed extraction")
    file_hash: str = Field(description="SHA256 hash of file")
    error_type: str = Field(description="Type of error (e.g., 'LLMError', 'ParseError', 'ValidationError')")
    error_message: str = Field(description="Full error message")
    attempts: int = Field(description="Number of extraction attempts made")
    models_used: List[str] = Field(description="List of models attempted (e.g., ['gpt-4o', 'gpt-4o-mini'])")
    failed_at: datetime = Field(default_factory=datetime.now, description="When extraction failed")
    prompt_version: str = Field(description="Prompt version used during extraction")

    def save_to_dlq(self, dlq_dir: Path) -> Path:
        """Save error metadata to Dead Letter Queue directory.

        Args:
            dlq_dir: Path to failed_extractions directory

        Returns:
            Path to saved metadata file
        """
        filename = f"{Path(self.file_path).stem}_{self.failed_at.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = dlq_dir / filename
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path


class LoadError(BaseModel):
    """Error metadata for failed graph loads (DLQ)."""

    json_path: str = Field(description="Path to JSON file that failed to load")
    file_hash: str = Field(description="SHA256 hash of source PDF")
    error_type: str = Field(description="Type of error (e.g., 'Neo4jError', 'ValidationError', 'IntegrityError')")
    error_message: str = Field(description="Full error message")
    failed_at: datetime = Field(default_factory=datetime.now, description="When load failed")
    partial_data_loaded: bool = Field(default=False, description="Whether some data was loaded before failure")

    def save_to_dlq(self, dlq_dir: Path) -> Path:
        """Save error metadata to Dead Letter Queue directory.

        Args:
            dlq_dir: Path to failed_loads directory

        Returns:
            Path to saved metadata file
        """
        filename = f"{Path(self.json_path).stem}_{self.failed_at.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = dlq_dir / filename
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path


class LoadResult(BaseModel):
    """Result of loading a PAF document into Neo4j."""

    success: bool = Field(description="Whether load was successful")
    document_id: str = Field(description="ID of loaded document")
    project_id: str = Field(description="ID of associated project")
    nodes_created: int = Field(default=0, description="Number of nodes created")
    relationships_created: int = Field(default=0, description="Number of relationships created")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    load_duration_seconds: float = Field(description="Time taken to load in seconds")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during load")
    loaded_at: datetime = Field(default_factory=datetime.now, description="When load completed")

    def get_summary(self) -> str:
        """Get human-readable load summary.

        Returns:
            Summary string with load statistics
        """
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return (
            f"{status} - Document: {self.document_id}\n"
            f"  Project: {self.project_id}\n"
            f"  Nodes created: {self.nodes_created}\n"
            f"  Relationships created: {self.relationships_created}\n"
            f"  Embeddings generated: {self.embeddings_generated}\n"
            f"  Duration: {self.load_duration_seconds:.2f}s\n"
            f"  Warnings: {len(self.warnings)}"
        )
