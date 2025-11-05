# PDF Documents Directory

Place your project PDF documents in this directory for ingestion and analysis.

## Supported Document Types

The system automatically detects document types based on file naming:

### PAF (Project Authorization Form)
- File naming: Include "PAF" in filename
- Example: `project_001_PAF.pdf`, `Alpha_PAF_v2.pdf`

### SRF (Status Report Form)
- File naming: Include "SRF" in filename
- Example: `project_002_SRF.pdf`, `Beta_SRF_Q3.pdf`

### IFR (Issue/Finding Report)
- File naming: Include "IFR" in filename
- Example: `project_003_IFR.pdf`, `Gamma_IFR_2024.pdf`

### Design Documents
- File naming: Include "Design" or "DESIGN" in filename
- Example: `System_Design_Doc.pdf`

### Specifications
- File naming: Include "Spec" or "SPEC" in filename
- Example: `Technical_Specifications.pdf`

### Other Documents
- Any PDF without specific keywords will be categorized as "Unknown"
- Will still be processed and indexed

## File Requirements

- **Format**: PDF (.pdf extension)
- **Text**: Must contain extractable text (not scanned images without OCR)
- **Size**: No strict limit, but files >50MB may take longer to process
- **Naming**: Use descriptive filenames for better organization

## Document Processing

Each PDF will be:
1. **Parsed**: Text extracted from all pages
2. **Chunked**: Split into overlapping chunks (~1000 characters)
3. **Indexed**: Stored in Neo4j with metadata
4. **Embedded**: Vector embeddings generated for semantic search
5. **Analyzed**: Entities and relationships extracted using LLM

## Sample Documents

For testing, you can create sample PDF documents or use the system with any project PDFs:

### Example Document Structure

**Project Authorization Form (PAF)**
- Project name and ID
- Project objectives
- Budget and timeline
- Approvals and signatures
- Related standards (e.g., "AACE-101", "ISO 9001")

**Status Report Form (SRF)**
- Project status update
- Progress metrics
- Team members involved
- Upcoming milestones

**Issue/Finding Report (IFR)**
- Issue description
- Risk level
- Affected projects
- Mitigation plans
- Referenced procedures

## Best Practices

1. **Organize by Project**: Create subdirectories for each project
   ```
   pdfs/
   ├── project_alpha/
   │   ├── PAF_001.pdf
   │   ├── SRF_Q1.pdf
   │   └── IFR_001.pdf
   └── project_beta/
       ├── PAF_002.pdf
       └── Design_Doc.pdf
   ```

2. **Use Consistent Naming**: Follow a naming convention
   - `{PROJECT}_{DOCTYPE}_{VERSION}.pdf`
   - Example: `ALPHA_PAF_v1.pdf`

3. **Include Metadata in Filenames**: Add dates or versions
   - `BETA_SRF_2024Q3.pdf`
   - `GAMMA_IFR_20240101.pdf`

4. **Keep PDFs Text-Based**: Scanned images require OCR preprocessing

## Processing Performance

- **Small PDFs** (1-10 pages): ~1-2 seconds per document
- **Medium PDFs** (10-50 pages): ~5-10 seconds per document
- **Large PDFs** (50+ pages): ~20-60 seconds per document

Batch processing of 100 documents takes approximately 5-15 minutes.

## Troubleshooting

### "Failed to parse PDF"
- Ensure PDF is not corrupted
- Try opening in a PDF reader
- Check file permissions

### "No text extracted"
- PDF may be scanned images
- Use OCR tool first: `ocrmypdf input.pdf output.pdf`

### "Encoding errors"
- Use UTF-8 compatible PDFs
- Re-save PDF with standard encoding

## Notes

- Subdirectories are automatically scanned
- PDF metadata (creation date, author) is not currently extracted
- Password-protected PDFs are not supported
- The system maintains original file paths for reference
