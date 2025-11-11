"""
Configuration management for GraphRAG Knowledge System.
Loads all settings from .env file using python-dotenv.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""

    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password is not empty."""
        if not v:
            raise ValueError("NEO4J_PASSWORD must be set in .env file")
        return v


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    fallback_model: str = Field(default_factory=lambda: os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"))

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is not empty."""
        if not v:
            raise ValueError("OPENAI_API_KEY must be set in .env file")
        return v


class DataConfig(BaseModel):
    """Data directory configuration."""

    pdf_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PDF_DATA_DIR", "./data/pdfs"))
    )
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    rejected_files_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("REJECTED_FILES_DIR", "./data/rejected_files"))
    )
    failed_extractions_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("FAILED_EXTRACTIONS_DIR", "./data/failed_extractions"))
    )
    failed_loads_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("FAILED_LOADS_DIR", "./data/failed_loads"))
    )

    @field_validator("pdf_data_dir", "output_dir", "rejected_files_dir", "failed_extractions_dir", "failed_loads_dir")
    @classmethod
    def validate_directories(cls, v: Path) -> Path:
        """Ensure directories exist or can be created."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    vector_dimension: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "384"))
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: Path = Field(default_factory=lambda: Path(os.getenv("LOG_FILE", "./logs/graphrag.log")))

    @field_validator("log_file")
    @classmethod
    def validate_log_file(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


class ExtractionConfig(BaseModel):
    """Configuration for PAF extraction pipeline."""

    prompt_version: str = Field(
        default_factory=lambda: os.getenv("EXTRACTION_PROMPT_VERSION", "v1.0.0")
    )
    enable_model_degradation: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_MODEL_DEGRADATION", "true").lower() == "true"
    )
    retry_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    )
    retry_backoff_base: int = Field(
        default_factory=lambda: int(os.getenv("RETRY_BACKOFF_BASE", "2"))
    )


class QualityConfig(BaseModel):
    """Configuration for quality thresholds and validation."""

    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "6.0"))
    )
    completeness_threshold: float = Field(
        default_factory=lambda: float(os.getenv("COMPLETENESS_THRESHOLD", "0.8"))
    )
    auto_generate_golden_tests: bool = Field(
        default_factory=lambda: os.getenv("AUTO_GENERATE_GOLDEN_TESTS", "true").lower() == "true"
    )


class Settings(BaseModel):
    """Main settings class combining all configurations."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
