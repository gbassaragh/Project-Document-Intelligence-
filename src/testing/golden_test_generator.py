"""
Golden test file generator for PAF extraction regression testing.

Generates golden test files from successfully extracted PAFs to ensure:
- Extraction quality remains consistent across prompt versions
- Schema changes don't break existing extractions
- Model upgrades maintain or improve quality
- Regression detection for confidence and completeness scores
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from difflib import unified_diff

from src.models.paf_document import PAFDocument
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class GoldenTestGenerator:
    """Generates and manages golden test files for extraction regression testing."""

    def __init__(self, golden_dir: Optional[Path] = None):
        """
        Initialize golden test generator.

        Args:
            golden_dir: Directory for golden test files (default: ./tests/golden_files)
        """
        self.settings = get_settings()
        self.golden_dir = golden_dir or Path("./tests/golden_files")
        self.golden_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file tracking all golden tests
        self.metadata_file = self.golden_dir / "golden_metadata.json"
        self.metadata = self._load_metadata()

        logger.info(f"GoldenTestGenerator initialized: {self.golden_dir}")

    def create_golden_test(
        self,
        paf_doc: PAFDocument,
        source_pdf_path: Path,
        notes: Optional[str] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Create golden test file from extracted PAF document.

        Args:
            paf_doc: Successfully extracted PAF document
            source_pdf_path: Path to source PDF file
            notes: Optional notes about this golden test
            overwrite: Whether to overwrite existing golden test (default: False)

        Returns:
            Path to created golden test file

        Raises:
            ValueError: If golden test already exists and overwrite=False
        """
        source_pdf_path = Path(source_pdf_path)

        # Generate golden test filename: {pdf_stem}_golden.json
        golden_filename = f"{source_pdf_path.stem}_golden.json"
        golden_path = self.golden_dir / golden_filename

        # Check if golden test already exists
        if golden_path.exists() and not overwrite:
            raise ValueError(
                f"Golden test already exists: {golden_filename}. "
                f"Use overwrite=True to replace."
            )

        # Copy source PDF to golden directory for reference
        pdf_copy_path = self.golden_dir / source_pdf_path.name
        if not pdf_copy_path.exists():
            shutil.copy2(source_pdf_path, pdf_copy_path)
            logger.debug(f"Copied source PDF to: {pdf_copy_path}")

        # Extract quality metrics
        quality_summary = paf_doc.get_quality_summary()

        # Create golden test record
        golden_record = {
            "source_pdf": source_pdf_path.name,
            "created_at": datetime.now().isoformat(),
            "prompt_version": paf_doc.prompt_version,
            "extraction_model": paf_doc.extraction_model,
            "notes": notes,
            "quality_baseline": {
                "average_confidence": quality_summary["average_confidence"],
                "completeness_score": quality_summary["completeness_score"],
                "required_fields_complete": quality_summary["required_fields_complete"],
                "optional_fields_present": quality_summary["optional_fields_present"],
                "financial_data_present": quality_summary["financial_data_present"],
                "revision_count": quality_summary["revision_count"]
            },
            "extracted_data": paf_doc.model_dump(mode='json')
        }

        # Save golden test file
        with open(golden_path, 'w', encoding='utf-8') as f:
            json.dump(golden_record, f, indent=2, default=str)

        logger.info(
            f"✅ Created golden test: {golden_filename} "
            f"(confidence={quality_summary['average_confidence']:.1f}, "
            f"completeness={quality_summary['completeness_score']:.2f})"
        )

        # Update metadata
        self._update_metadata(
            golden_filename=golden_filename,
            source_pdf=source_pdf_path.name,
            prompt_version=paf_doc.prompt_version,
            quality_metrics=quality_summary
        )

        return golden_path

    def compare_against_golden(
        self,
        paf_doc: PAFDocument,
        golden_filename: str
    ) -> Dict[str, any]:
        """
        Compare extracted PAF document against golden test.

        Args:
            paf_doc: Newly extracted PAF document
            golden_filename: Name of golden test file to compare against

        Returns:
            Comparison report with differences and quality deltas

        Raises:
            FileNotFoundError: If golden test file doesn't exist
        """
        golden_path = self.golden_dir / golden_filename

        if not golden_path.exists():
            raise FileNotFoundError(f"Golden test not found: {golden_filename}")

        # Load golden test
        with open(golden_path, 'r', encoding='utf-8') as f:
            golden_record = json.load(f)

        golden_data = golden_record["extracted_data"]
        golden_quality = golden_record["quality_baseline"]

        # Current document metrics
        current_quality = paf_doc.get_quality_summary()

        # Compare quality metrics
        quality_delta = {
            "average_confidence_delta": (
                current_quality["average_confidence"] -
                golden_quality["average_confidence"]
            ),
            "completeness_score_delta": (
                current_quality["completeness_score"] -
                golden_quality["completeness_score"]
            ),
            "optional_fields_delta": (
                current_quality["optional_fields_present"] -
                golden_quality["optional_fields_present"]
            ),
            "financial_data_delta": (
                current_quality["financial_data_present"] -
                golden_quality["financial_data_present"]
            )
        }

        # Compare field values
        field_differences = self._compare_fields(
            current=paf_doc.model_dump(mode='json'),
            golden=golden_data
        )

        # Determine test result
        passed = (
            len(field_differences) == 0 and
            quality_delta["average_confidence_delta"] >= -1.0 and
            quality_delta["completeness_score_delta"] >= -0.05
        )

        comparison_report = {
            "golden_file": golden_filename,
            "golden_prompt_version": golden_record["prompt_version"],
            "current_prompt_version": paf_doc.prompt_version,
            "test_passed": passed,
            "quality_delta": quality_delta,
            "field_differences": field_differences,
            "field_difference_count": len(field_differences),
            "compared_at": datetime.now().isoformat()
        }

        # Log results
        if passed:
            logger.info(
                f"✅ Golden test PASSED: {golden_filename} "
                f"(confidence_delta={quality_delta['average_confidence_delta']:+.1f}, "
                f"completeness_delta={quality_delta['completeness_score_delta']:+.2f})"
            )
        else:
            logger.warning(
                f"⚠️ Golden test FAILED: {golden_filename} "
                f"({len(field_differences)} field differences, "
                f"confidence_delta={quality_delta['average_confidence_delta']:+.1f})"
            )

        return comparison_report

    def _compare_fields(
        self,
        current: Dict,
        golden: Dict,
        path: str = ""
    ) -> List[Dict[str, any]]:
        """
        Recursively compare field values between current and golden extractions.

        Args:
            current: Current extraction data
            golden: Golden test data
            path: Current field path (for nested fields)

        Returns:
            List of field differences
        """
        differences = []

        # Compare required fields (ignore metadata fields)
        ignore_fields = {
            "extraction_timestamp", "file_hash", "extraction_model",
            "prompt_version", "completeness_score"
        }

        all_keys = set(current.keys()) | set(golden.keys())
        for key in all_keys:
            if key in ignore_fields:
                continue

            field_path = f"{path}.{key}" if path else key

            # Missing field
            if key not in current:
                differences.append({
                    "field": field_path,
                    "type": "missing_in_current",
                    "golden_value": golden.get(key),
                    "current_value": None
                })
            elif key not in golden:
                differences.append({
                    "field": field_path,
                    "type": "missing_in_golden",
                    "golden_value": None,
                    "current_value": current.get(key)
                })
            else:
                current_val = current[key]
                golden_val = golden[key]

                # Recursive comparison for nested dicts
                if isinstance(current_val, dict) and isinstance(golden_val, dict):
                    nested_diffs = self._compare_fields(current_val, golden_val, field_path)
                    differences.extend(nested_diffs)
                # List comparison
                elif isinstance(current_val, list) and isinstance(golden_val, list):
                    if len(current_val) != len(golden_val):
                        differences.append({
                            "field": field_path,
                            "type": "list_length_mismatch",
                            "golden_value": f"length={len(golden_val)}",
                            "current_value": f"length={len(current_val)}"
                        })
                # Value comparison (with tolerance for floats)
                elif current_val != golden_val:
                    # Float comparison with tolerance
                    if isinstance(current_val, (int, float)) and isinstance(golden_val, (int, float)):
                        if abs(float(current_val) - float(golden_val)) > 0.01:
                            differences.append({
                                "field": field_path,
                                "type": "value_mismatch",
                                "golden_value": golden_val,
                                "current_value": current_val
                            })
                    # String comparison (case-insensitive for text fields)
                    elif isinstance(current_val, str) and isinstance(golden_val, str):
                        if current_val.strip().lower() != golden_val.strip().lower():
                            differences.append({
                                "field": field_path,
                                "type": "value_mismatch",
                                "golden_value": golden_val,
                                "current_value": current_val
                            })
                    else:
                        differences.append({
                            "field": field_path,
                            "type": "value_mismatch",
                            "golden_value": golden_val,
                            "current_value": current_val
                        })

        return differences

    def run_all_golden_tests(
        self,
        extractor,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Run all golden tests and generate regression report.

        Args:
            extractor: PAFExtractor instance for re-extraction
            verbose: Whether to print detailed differences

        Returns:
            Regression test report with pass/fail status
        """
        logger.info("Running all golden tests...")

        results = []
        passed_count = 0
        failed_count = 0

        for golden_file in self.golden_dir.glob("*_golden.json"):
            # Load golden test
            with open(golden_file, 'r', encoding='utf-8') as f:
                golden_record = json.load(f)

            source_pdf_name = golden_record["source_pdf"]
            source_pdf_path = self.golden_dir / source_pdf_name

            if not source_pdf_path.exists():
                logger.warning(f"Source PDF not found for golden test: {source_pdf_name}")
                continue

            # Re-extract PDF
            try:
                paf_doc = extractor.extract_with_retry(source_pdf_path)

                # Compare against golden
                comparison = self.compare_against_golden(paf_doc, golden_file.name)
                results.append(comparison)

                if comparison["test_passed"]:
                    passed_count += 1
                else:
                    failed_count += 1

                    if verbose:
                        logger.info(f"\nDifferences in {golden_file.name}:")
                        for diff in comparison["field_differences"]:
                            logger.info(
                                f"  {diff['field']}: "
                                f"golden={diff['golden_value']} "
                                f"current={diff['current_value']}"
                            )

            except Exception as e:
                logger.error(f"Failed to re-extract {source_pdf_name}: {str(e)}")
                failed_count += 1
                results.append({
                    "golden_file": golden_file.name,
                    "test_passed": False,
                    "error": str(e)
                })

        # Generate summary report
        report = {
            "total_tests": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": (passed_count / len(results) * 100) if results else 0,
            "results": results,
            "run_at": datetime.now().isoformat()
        }

        logger.info(
            f"\n{'='*60}\n"
            f"Golden Test Summary:\n"
            f"  Total: {report['total_tests']}\n"
            f"  Passed: {passed_count} ✅\n"
            f"  Failed: {failed_count} ❌\n"
            f"  Pass Rate: {report['pass_rate']:.1f}%\n"
            f"{'='*60}"
        )

        return report

    def _load_metadata(self) -> Dict:
        """Load golden test metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"golden_tests": {}, "last_updated": None}

    def _update_metadata(
        self,
        golden_filename: str,
        source_pdf: str,
        prompt_version: str,
        quality_metrics: Dict
    ) -> None:
        """Update golden test metadata."""
        self.metadata["golden_tests"][golden_filename] = {
            "source_pdf": source_pdf,
            "prompt_version": prompt_version,
            "quality_metrics": quality_metrics,
            "created_at": datetime.now().isoformat()
        }
        self.metadata["last_updated"] = datetime.now().isoformat()

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def list_golden_tests(self) -> List[Dict[str, any]]:
        """
        List all available golden tests with metadata.

        Returns:
            List of golden test summaries
        """
        golden_tests = []

        for golden_file in self.golden_dir.glob("*_golden.json"):
            with open(golden_file, 'r', encoding='utf-8') as f:
                golden_record = json.load(f)

            golden_tests.append({
                "filename": golden_file.name,
                "source_pdf": golden_record["source_pdf"],
                "prompt_version": golden_record["prompt_version"],
                "extraction_model": golden_record["extraction_model"],
                "created_at": golden_record["created_at"],
                "quality_baseline": golden_record["quality_baseline"]
            })

        return sorted(golden_tests, key=lambda x: x["created_at"], reverse=True)

    def delete_golden_test(self, golden_filename: str) -> None:
        """
        Delete golden test file and update metadata.

        Args:
            golden_filename: Name of golden test file to delete
        """
        golden_path = self.golden_dir / golden_filename

        if not golden_path.exists():
            raise FileNotFoundError(f"Golden test not found: {golden_filename}")

        golden_path.unlink()
        logger.info(f"Deleted golden test: {golden_filename}")

        # Update metadata
        if golden_filename in self.metadata["golden_tests"]:
            del self.metadata["golden_tests"][golden_filename]
            self.metadata["last_updated"] = datetime.now().isoformat()

            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
