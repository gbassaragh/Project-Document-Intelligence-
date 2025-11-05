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


class FileMapping(BaseModel):
    """Mapping configuration for a single structured data file."""

    table_name: str = Field(description="Target table name in DuckDB")
    entity_type: str = Field(description="Entity type for Neo4j ingestion (e.g., 'Person', 'Project')")
    required_columns: List[str] = Field(
        default_factory=list,
        description="List of required column names"
    )


class StructuredDataIngestionConfig(BaseModel):
    """Configuration for structured data ingestion pipeline.

    Replaces brittle heuristic-based table discovery with explicit file-to-table mapping.
    """

    file_mappings: Dict[str, FileMapping] = Field(
        default_factory=lambda: {
            # Default mappings for common file patterns
            "projects.xlsx": FileMapping(
                table_name="projects",
                entity_type="Project",
                required_columns=["id", "name"]
            ),
            "projects.csv": FileMapping(
                table_name="projects",
                entity_type="Project",
                required_columns=["id", "name"]
            ),
            "managers.xlsx": FileMapping(
                table_name="managers",
                entity_type="Person",
                required_columns=["name"]
            ),
            "managers.csv": FileMapping(
                table_name="managers",
                entity_type="Person",
                required_columns=["name"]
            ),
            "teams.xlsx": FileMapping(
                table_name="teams",
                entity_type="Team",
                required_columns=["team_name", "member_name"]
            ),
            "teams.csv": FileMapping(
                table_name="teams",
                entity_type="Team",
                required_columns=["team_name", "member_name"]
            ),
        },
        description="Mapping from file names to their DuckDB table configuration"
    )


class Settings(BaseModel):
    """Main settings class combining all configurations."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    structured_data_ingestion: StructuredDataIngestionConfig = Field(
        default_factory=StructuredDataIngestionConfig
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
