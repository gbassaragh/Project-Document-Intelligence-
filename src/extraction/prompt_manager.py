"""
Prompt manager for versioned extraction prompts.

Manages loading, versioning, and tracking of extraction prompts across
the system. Enables selective re-extraction when prompts are updated.
"""

import logging
from pathlib import Path
from typing import Optional
import re

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages versioned extraction prompts."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Directory containing prompt templates (default: ./prompts)
        """
        self.settings = get_settings()
        self.prompts_dir = prompts_dir or Path("./prompts")
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def load_prompt(self, version: str) -> str:
        """
        Load prompt template by version.

        Args:
            version: Prompt version (e.g., 'v1.0.0', 'v1.1.0')

        Returns:
            Prompt template text

        Raises:
            FileNotFoundError: If prompt version not found
            ValueError: If prompt file is empty
        """
        # Clean version string (remove 'v' prefix if present)
        version_clean = version.lstrip('v')

        # Try with and without 'v' prefix
        possible_names = [
            f"v{version_clean}_paf_extraction.txt",
            f"{version_clean}_paf_extraction.txt",
        ]

        for filename in possible_names:
            prompt_path = self.prompts_dir / filename
            if prompt_path.exists():
                logger.debug(f"Loading prompt template: {prompt_path.name}")

                content = prompt_path.read_text()
                if not content.strip():
                    raise ValueError(f"Prompt file is empty: {prompt_path}")

                logger.info(f"Loaded prompt template: {version} ({len(content)} chars)")
                return content

        # If not found, raise error
        raise FileNotFoundError(
            f"Prompt version '{version}' not found in {self.prompts_dir}. "
            f"Tried: {', '.join(possible_names)}"
        )

    def get_current_version(self) -> str:
        """
        Get current prompt version from configuration.

        Returns:
            Current prompt version (e.g., 'v1.0.0')
        """
        return self.settings.extraction.prompt_version

    def load_current_prompt(self) -> str:
        """
        Load the current prompt version from configuration.

        Returns:
            Current prompt template text

        Raises:
            FileNotFoundError: If current prompt version not found
        """
        current_version = self.get_current_version()
        return self.load_prompt(current_version)

    def list_available_versions(self) -> list[str]:
        """
        List all available prompt versions in the prompts directory.

        Returns:
            List of version strings sorted by semantic version
        """
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory does not exist: {self.prompts_dir}")
            return []

        # Pattern: v1.0.0_paf_extraction.txt or 1.0.0_paf_extraction.txt
        pattern = re.compile(r"^v?(\d+\.\d+\.\d+)_paf_extraction\.txt$")

        versions = []
        for file_path in self.prompts_dir.glob("*_paf_extraction.txt"):
            match = pattern.match(file_path.name)
            if match:
                version = f"v{match.group(1)}"
                versions.append(version)

        # Sort by semantic version
        versions.sort(key=lambda v: [int(x) for x in v.lstrip('v').split('.')])

        logger.debug(f"Found {len(versions)} prompt versions: {', '.join(versions)}")
        return versions

    def validate_prompt(self, prompt_text: str) -> bool:
        """
        Validate prompt template has required components.

        Checks for:
        - Minimum length (> 100 characters)
        - Required field instructions
        - Output format specification
        - Confidence scoring instructions

        Args:
            prompt_text: Prompt template text

        Returns:
            True if valid, False otherwise
        """
        if len(prompt_text) < 100:
            logger.warning("Prompt is too short (< 100 characters)")
            return False

        required_keywords = [
            "project_number",
            "project_title",
            "company",
            "scope_text",
            "cost_text",
            "confidence",
            "JSON"
        ]

        missing_keywords = [kw for kw in required_keywords if kw.lower() not in prompt_text.lower()]

        if missing_keywords:
            logger.warning(f"Prompt missing required keywords: {', '.join(missing_keywords)}")
            return False

        logger.debug("Prompt validation passed")
        return True

    def create_prompt_version(
        self,
        version: str,
        prompt_text: str,
        overwrite: bool = False
    ) -> Path:
        """
        Create a new prompt version file.

        Args:
            version: Version string (e.g., 'v1.1.0')
            prompt_text: Prompt template text
            overwrite: Whether to overwrite existing version (default: False)

        Returns:
            Path to created prompt file

        Raises:
            ValueError: If prompt is invalid or version exists
        """
        if not self.validate_prompt(prompt_text):
            raise ValueError("Prompt validation failed")

        # Clean version string
        version_clean = version.lstrip('v')
        filename = f"v{version_clean}_paf_extraction.txt"
        prompt_path = self.prompts_dir / filename

        if prompt_path.exists() and not overwrite:
            raise ValueError(f"Prompt version '{version}' already exists (use overwrite=True to replace)")

        prompt_path.write_text(prompt_text)
        logger.info(f"Created prompt version: {filename}")

        return prompt_path

    def compare_versions(self, version1: str, version2: str) -> dict:
        """
        Compare two prompt versions and report differences.

        Args:
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary with comparison results
        """
        prompt1 = self.load_prompt(version1)
        prompt2 = self.load_prompt(version2)

        # Basic comparison metrics
        return {
            "version1": version1,
            "version2": version2,
            "length_diff": len(prompt2) - len(prompt1),
            "length_diff_percent": ((len(prompt2) - len(prompt1)) / len(prompt1)) * 100,
            "identical": prompt1 == prompt2,
        }

    def get_prompt_metadata(self, version: str) -> dict:
        """
        Get metadata about a prompt version.

        Args:
            version: Prompt version

        Returns:
            Dictionary with metadata
        """
        prompt_text = self.load_prompt(version)

        version_clean = version.lstrip('v')
        filename = f"v{version_clean}_paf_extraction.txt"
        prompt_path = self.prompts_dir / filename

        stat = prompt_path.stat()

        return {
            "version": version,
            "file_path": str(prompt_path),
            "file_size": stat.st_size,
            "modified_at": stat.st_mtime,
            "line_count": len(prompt_text.split('\n')),
            "char_count": len(prompt_text),
            "is_valid": self.validate_prompt(prompt_text),
        }
