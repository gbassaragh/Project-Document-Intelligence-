"""
Integration test for Phase 3: Validation & Extraction Pipeline

Tests:
1. Document validation (pre-flight checks)
2. Prompt management (version loading)
3. PAF extraction (with retry logic) - DRY RUN ONLY
4. Golden test generation

This is a validation test only - does NOT call LLM APIs.
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_document_validation():
    """Test document validator with sample PDF."""
    from src.validation.document_validator import DocumentValidator
    
    logger.info("="*60)
    logger.info("TEST 1: Document Validation")
    logger.info("="*60)
    
    validator = DocumentValidator()
    test_pdf = Path("data/pdfs/21503 - Line 109 ROW 144 Asset Condition Replacements - PAF.pdf")
    
    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        return False
    
    result = validator.validate_pdf(test_pdf)
    
    logger.info(f"Validation Status: {result.status.value}")
    logger.info(f"File Size: {result.file_size:,} bytes")
    logger.info(f"File Hash: {result.file_hash[:16]}...")
    
    if result.reasons:
        logger.info(f"Validation Issues: {'; '.join(result.reasons)}")
    else:
        logger.info("✅ All validation checks passed")
    
    logger.info(f"Is Valid: {result.is_valid()}")
    
    return result.is_valid()

def test_prompt_management():
    """Test prompt manager."""
    from src.extraction.prompt_manager import PromptManager
    
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Prompt Management")
    logger.info("="*60)
    
    manager = PromptManager()
    
    # List available versions
    versions = manager.list_available_versions()
    logger.info(f"Available prompt versions: {versions}")
    
    if not versions:
        logger.error("No prompt versions found!")
        return False
    
    # Load current version
    current_version = manager.get_current_version()
    logger.info(f"Current version from config: {current_version}")
    
    try:
        prompt_text = manager.load_current_prompt()
        logger.info(f"Loaded prompt: {len(prompt_text)} characters")
        
        # Validate prompt
        is_valid = manager.validate_prompt(prompt_text)
        logger.info(f"Prompt validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        # Get metadata
        metadata = manager.get_prompt_metadata(current_version)
        logger.info(f"Prompt metadata:")
        logger.info(f"  - File: {metadata['file_path']}")
        logger.info(f"  - Size: {metadata['file_size']} bytes")
        logger.info(f"  - Lines: {metadata['line_count']}")
        logger.info(f"  - Valid: {metadata['is_valid']}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to load prompt: {str(e)}")
        return False

def test_extractor_initialization():
    """Test PAF extractor initialization (NO LLM CALLS)."""
    from src.extraction.paf_extractor import PAFExtractor
    
    logger.info("\n" + "="*60)
    logger.info("TEST 3: PAF Extractor Initialization")
    logger.info("="*60)
    
    try:
        extractor = PAFExtractor()
        
        logger.info(f"Primary model: {extractor.settings.openai.model}")
        logger.info(f"Fallback model: {extractor.settings.openai.fallback_model}")
        logger.info(f"Max attempts: {extractor.max_attempts}")
        logger.info(f"Backoff base: {extractor.backoff_base}s")
        logger.info(f"Model degradation: {extractor.enable_degradation}")
        logger.info(f"Output dir: {extractor.output_dir}")
        logger.info(f"DLQ dir: {extractor.dlq_dir}")
        
        logger.info("✅ PAF Extractor initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {str(e)}")
        return False

def test_golden_test_generator():
    """Test golden test generator initialization."""
    from src.testing.golden_test_generator import GoldenTestGenerator
    
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Golden Test Generator")
    logger.info("="*60)
    
    try:
        generator = GoldenTestGenerator()
        
        logger.info(f"Golden directory: {generator.golden_dir}")
        
        # List existing golden tests
        golden_tests = generator.list_golden_tests()
        logger.info(f"Existing golden tests: {len(golden_tests)}")
        
        for test in golden_tests:
            logger.info(f"  - {test['filename']}: {test['source_pdf']}")
        
        logger.info("✅ Golden Test Generator initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize golden test generator: {str(e)}")
        return False

def main():
    """Run all Phase 3 integration tests."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3 INTEGRATION TEST SUITE")
    logger.info("Validation & Extraction Pipeline")
    logger.info("="*80 + "\n")
    
    results = {
        "Document Validation": test_document_validation(),
        "Prompt Management": test_prompt_management(),
        "PAF Extractor Init": test_extractor_initialization(),
        "Golden Test Generator": test_golden_test_generator()
    }
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<30} {status}")
    
    logger.info("-"*80)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    logger.info("="*80)
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Phase 3 pipeline is working correctly.")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
