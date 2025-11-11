"""
Pydantic models for PAF (Project Authorization Form) document extraction.

These models define the complete structure for extracted PAF data including:
- Header fields (project number, title, company, people)
- Content sections (scope, cost)
- Financial metadata
- Revision history
- Quality tracking (confidence scores, completeness)
- Extraction metadata (prompt version, model, timestamp)
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Revision(BaseModel):
    """Document revision metadata."""

    version: str = Field(description="Revision version (e.g., 'Rev 1', 'Rev 2', 'Rev A')")
    date: Optional[str] = Field(None, description="Revision date (YYYY-MM-DD format preferred)")
    description: Optional[str] = Field(None, description="Brief description of changes")


class ConfidenceScores(BaseModel):
    """Confidence scores for extracted fields (1-10 scale).

    Confidence scoring guidelines:
    - 10: Field explicitly present with clear label, no ambiguity
    - 7-9: Field present but label unclear or value requires interpretation
    - 4-6: Field possibly present but significant uncertainty
    - 1-3: Field not found or highly ambiguous
    """

    project_number: int = Field(ge=1, le=10, description="Confidence in project number extraction")
    project_title: int = Field(ge=1, le=10, description="Confidence in project title extraction")
    company: int = Field(ge=1, le=10, description="Confidence in company extraction")
    project_manager: int = Field(ge=1, le=10, description="Confidence in project manager extraction")
    project_sponsor: int = Field(ge=1, le=10, description="Confidence in project sponsor extraction")
    project_initiator: int = Field(ge=1, le=10, description="Confidence in project initiator extraction")
    scope_text: int = Field(ge=1, le=10, description="Confidence in scope section extraction")
    cost_text: int = Field(ge=1, le=10, description="Confidence in cost section extraction")


class CostBreakdown(BaseModel):
    """Enhanced financial metadata extracted from Cost and Funding section."""

    total_request: Optional[float] = Field(None, description="Total capital request amount")
    year_1_cost: Optional[float] = Field(None, description="First year expenditure")
    year_2_cost: Optional[float] = Field(None, description="Second year expenditure")
    year_3_cost: Optional[float] = Field(None, description="Third year expenditure")
    contingency_percent: Optional[float] = Field(None, ge=0, le=100, description="Contingency percentage")
    funding_source: Optional[str] = Field(
        None,
        description="Funding source classification: 'internal', 'external', or 'mixed'"
    )

    def validate_funding_source(self) -> None:
        """Validate funding source is one of the allowed values."""
        if self.funding_source is not None:
            allowed = ["internal", "external", "mixed"]
            if self.funding_source.lower() not in allowed:
                raise ValueError(f"funding_source must be one of {allowed}, got: {self.funding_source}")


class PAFDocument(BaseModel):
    """Complete PAF document extraction result with quality tracking."""

    # ===== HEADER FIELDS (REQUIRED) =====
    project_number: str = Field(description="Unique project identifier")
    project_title: str = Field(description="Full project title/name")
    company: str = Field(description="Operating company name (e.g., 'Eversource NH', 'NSTAR Electric')")

    # ===== HEADER FIELDS (OPTIONAL) =====
    project_manager: Optional[str] = Field(None, description="Name of project manager")
    project_sponsor: Optional[str] = Field(None, description="Name of executive sponsor")
    project_initiator: Optional[str] = Field(None, description="Name of project initiator")

    # ===== CONTENT SECTIONS (REQUIRED) =====
    scope_text: str = Field(description="Full text of [3] Project Scope section")
    cost_text: str = Field(description="Full text of [5] Cost and Funding section")

    # ===== FINANCIAL DATA =====
    cost_breakdown: CostBreakdown = Field(description="Extracted financial metadata")

    # ===== REVISION HISTORY =====
    revisions: List[Revision] = Field(default_factory=list, description="Document revision history")

    # ===== QUALITY METADATA =====
    confidence_scores: ConfidenceScores = Field(description="Per-field confidence scores (1-10 scale)")
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of fields successfully extracted (0.0 - 1.0)"
    )

    # ===== EXTRACTION METADATA =====
    prompt_version: str = Field(description="Extraction prompt version (e.g., 'v1.0.0')")
    extraction_model: str = Field(description="Model used for extraction ('gpt-4o' or 'gpt-4o-mini')")
    extraction_timestamp: datetime = Field(description="When extraction was performed")
    file_hash: str = Field(description="SHA256 hash of source PDF for duplicate detection")

    def calculate_average_confidence(self) -> float:
        """Calculate average confidence across all fields.

        Returns:
            Average confidence score (1.0 - 10.0)
        """
        scores = self.confidence_scores.dict().values()
        return sum(scores) / len(scores)

    def is_required_fields_complete(self) -> bool:
        """Check if all required fields are present and non-empty.

        Returns:
            True if all required fields have values
        """
        return bool(
            self.project_number and
            self.project_title and
            self.company and
            self.scope_text and
            self.cost_text
        )

    def calculate_completeness_score(self) -> float:
        """Calculate completeness score based on field presence.

        Weighting:
        - Required fields (weight 1.0): project_number, project_title, company, scope_text, cost_text
        - Optional fields (weight 0.5): manager, sponsor, initiator, financial breakdown items

        Returns:
            Completeness score (0.0 - 1.0)
        """
        # Required fields (5 fields, weight 1.0 each)
        required_score = 0.0
        required_fields = [
            self.project_number,
            self.project_title,
            self.company,
            self.scope_text,
            self.cost_text
        ]
        required_score = sum(1.0 for field in required_fields if field) / len(required_fields)

        # Optional fields (8 fields, weight 0.5 each)
        optional_score = 0.0
        optional_fields = [
            self.project_manager,
            self.project_sponsor,
            self.project_initiator,
            self.cost_breakdown.total_request,
            self.cost_breakdown.year_1_cost,
            self.cost_breakdown.year_2_cost,
            self.cost_breakdown.year_3_cost,
            self.cost_breakdown.contingency_percent,
        ]
        optional_score = sum(0.5 for field in optional_fields if field is not None) / len(optional_fields)

        # Combined score (required fields weighted more heavily)
        return (required_score * 0.7) + (optional_score * 0.3)

    def get_quality_summary(self) -> dict:
        """Get a summary of extraction quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        return {
            "average_confidence": round(self.calculate_average_confidence(), 2),
            "completeness_score": round(self.completeness_score, 2),
            "required_fields_complete": self.is_required_fields_complete(),
            "optional_fields_present": sum(1 for field in [
                self.project_manager,
                self.project_sponsor,
                self.project_initiator
            ] if field is not None),
            "financial_data_present": sum(1 for field in [
                self.cost_breakdown.total_request,
                self.cost_breakdown.year_1_cost,
                self.cost_breakdown.year_2_cost,
                self.cost_breakdown.year_3_cost,
            ] if field is not None),
            "revision_count": len(self.revisions),
        }

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "project_number": "NH-2024-001",
                "project_title": "Substation Upgrade Project",
                "company": "Eversource NH",
                "project_manager": "John Smith",
                "project_sponsor": "Jane Doe",
                "project_initiator": "Bob Johnson",
                "scope_text": "This project involves upgrading the existing substation...",
                "cost_text": "Total capital request: $2.5M. Year 1: $1.5M, Year 2: $1.0M...",
                "cost_breakdown": {
                    "total_request": 2500000.0,
                    "year_1_cost": 1500000.0,
                    "year_2_cost": 1000000.0,
                    "year_3_cost": None,
                    "contingency_percent": 15.0,
                    "funding_source": "internal"
                },
                "revisions": [
                    {"version": "Rev 1", "date": "2024-01-15", "description": "Initial submission"},
                    {"version": "Rev 2", "date": "2024-02-20", "description": "Updated cost estimates"}
                ],
                "confidence_scores": {
                    "project_number": 10,
                    "project_title": 10,
                    "company": 10,
                    "project_manager": 9,
                    "project_sponsor": 8,
                    "project_initiator": 7,
                    "scope_text": 9,
                    "cost_text": 9
                },
                "completeness_score": 0.95,
                "prompt_version": "v1.0.0",
                "extraction_model": "gpt-4o",
                "extraction_timestamp": "2024-11-10T12:00:00",
                "file_hash": "abc123..."
            }
        }
