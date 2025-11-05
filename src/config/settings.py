"""
Configuration management for GraphRAG Knowledge System.
Loads all settings from .env file using python-dotenv.
"""

import os
from pathlib import Path
from typing import Optional

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

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key is not empty."""
        if not v:
            raise ValueError("OPENAI_API_KEY must be set in .env file")
        return v


class DataConfig(BaseModel):
    """Data directory configuration."""

    structured_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("STRUCTURED_DATA_DIR", "./data/structured"))
    )
    pdf_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PDF_DATA_DIR", "./data/pdfs"))
    )
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))

    @field_validator("structured_data_dir", "pdf_data_dir", "output_dir")
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


class Settings(BaseModel):
    """Main settings class combining all configurations."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
