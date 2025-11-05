#!/usr/bin/env python3
"""
Installation verification script for GraphRAG Knowledge System.
Run this script to verify all components are properly configured.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)

    required_packages = [
        "dotenv",
        "neo4j",
        "duckdb",
        "pandas",
        "pypdf",
        "langchain",
        "langchain_openai",
        "sentence_transformers",
        "pydantic",
        "tqdm",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed")
    return True


def check_configuration():
    """Check if configuration is properly set."""
    print("\n" + "=" * 80)
    print("CHECKING CONFIGURATION")
    print("=" * 80)

    try:
        from src.config.settings import get_settings

        settings = get_settings()

        # Check Neo4j configuration
        print(f"✅ Neo4j URI: {settings.neo4j.uri}")
        print(f"✅ Neo4j Username: {settings.neo4j.username}")
        print(f"✅ Neo4j Database: {settings.neo4j.database}")

        # Check OpenAI configuration
        api_key = settings.openai.api_key
        if api_key and len(api_key) > 20:
            print(f"✅ OpenAI API Key: {api_key[:20]}...")
            print(f"✅ OpenAI Model: {settings.openai.model}")
        else:
            print("❌ OpenAI API Key: NOT SET")
            return False

        # Check data directories
        print(f"✅ Structured Data Dir: {settings.data.structured_data_dir}")
        print(f"✅ PDF Data Dir: {settings.data.pdf_data_dir}")
        print(f"✅ Output Dir: {settings.data.output_dir}")

        print("\n✅ Configuration loaded successfully")
        return True

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure .env file exists and contains:")
        print("  - NEO4J_PASSWORD")
        print("  - OPENAI_API_KEY")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_neo4j_connection():
    """Check Neo4j database connection."""
    print("\n" + "=" * 80)
    print("CHECKING NEO4J CONNECTION")
    print("=" * 80)

    try:
        from src.database.connection import get_connection

        connection = get_connection()

        if connection.verify_connection():
            print("✅ Neo4j connection successful")

            # Get database info
            query = "CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version"
            result = connection.execute_query(query)
            if result:
                print(f"✅ Neo4j Version: {result[0]['version']}")

            return True
        else:
            print("❌ Failed to connect to Neo4j")
            return False

    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify Neo4j is running: docker ps | grep neo4j")
        print("  2. Check NEO4J_URI in .env (default: bolt://localhost:7687)")
        print("  3. Verify NEO4J_PASSWORD matches your Neo4j password")
        return False


def check_data_directories():
    """Check if data directories exist and have content."""
    print("\n" + "=" * 80)
    print("CHECKING DATA DIRECTORIES")
    print("=" * 80)

    from src.config.settings import get_settings

    settings = get_settings()

    # Check structured data
    structured_files = list(settings.data.structured_data_dir.glob("*.xlsx")) + list(
        settings.data.structured_data_dir.glob("*.csv")
    )
    print(
        f"{'✅' if structured_files else '⚠️'} Structured data files: {len(structured_files)}"
    )

    # Check PDF files
    pdf_files = list(settings.data.pdf_data_dir.glob("**/*.pdf"))
    print(f"{'✅' if pdf_files else '⚠️'} PDF files: {len(pdf_files)}")

    if not structured_files and not pdf_files:
        print("\n⚠️  No data files found")
        print("Add Excel/CSV files to: data/structured/")
        print("Add PDF files to: data/pdfs/")
        print("Sample files are provided in the directories")
        return False

    return True


def check_schema():
    """Check if Neo4j schema is initialized."""
    print("\n" + "=" * 80)
    print("CHECKING NEO4J SCHEMA")
    print("=" * 80)

    try:
        from src.database.connection import get_connection
        from src.database.schema import SchemaManager

        connection = get_connection()
        schema_manager = SchemaManager(connection)

        schema_info = schema_manager.get_schema_info()

        constraints = schema_info.get("constraints", [])
        indexes = schema_info.get("indexes", [])

        print(f"✅ Constraints: {len(constraints)}")
        print(f"✅ Indexes: {len(indexes)}")

        if constraints:
            for constraint in constraints[:3]:
                print(f"   - {constraint.get('name', 'Unknown')}")

        if len(constraints) < 5:
            print(
                "\n⚠️  Schema not fully initialized. Run: python main.py -> Option 2"
            )
            return False

        return True

    except Exception as e:
        print(f"❌ Schema check error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("GraphRAG Knowledge System - Installation Verification")
    print("=" * 80)

    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
        ("Neo4j Connection", check_neo4j_connection),
        ("Data Directories", check_data_directories),
        ("Neo4j Schema", check_schema),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Your system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Select Option 1 to run full ingestion pipeline")
        print("  3. Select Option 8 for interactive query mode")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("See QUICKSTART.md for troubleshooting guide")

    print()


if __name__ == "__main__":
    main()
