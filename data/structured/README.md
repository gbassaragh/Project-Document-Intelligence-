# Structured Data Directory

This directory contains Excel (.xlsx) and CSV (.csv) files with structured project data.

## Expected File Formats

### projects.xlsx / projects.csv
Project information with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | String | Unique project identifier | PROJ-001 |
| name | String | Project name | Alpha Construction |
| status | String | Project status | Active, Completed, On Hold |
| manager | String (Optional) | Project manager name | John Doe |

**Example:**
```csv
id,name,status,manager
PROJ-001,Alpha Construction,Active,John Doe
PROJ-002,Beta Infrastructure,Completed,Jane Smith
PROJ-003,Gamma Development,On Hold,Bob Johnson
```

### teams.xlsx / teams.csv
Team membership information:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| team_name | String | Team name | Engineering Team |
| member_name | String | Team member name | Alice Johnson |

**Example:**
```csv
team_name,member_name
Engineering Team,John Doe
Engineering Team,Alice Johnson
Design Team,Jane Smith
Design Team,Bob Johnson
```

### managers.xlsx / managers.csv (Optional)
Project manager assignments:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| project_id | String | Project ID | PROJ-001 |
| manager_name | String | Manager name | John Doe |

**Example:**
```csv
project_id,manager_name
PROJ-001,John Doe
PROJ-002,Jane Smith
PROJ-003,Bob Johnson
```

## Data Validation Rules

1. **Unique Identifiers**: Project IDs must be unique
2. **Required Fields**: All core fields (id, name, team_name, member_name) must be non-null
3. **Consistent Naming**: Use consistent person names across all files
4. **Status Values**: Recommended status values: Active, Completed, On Hold, Cancelled, Planned

## Loading Process

The system automatically:
1. Scans this directory for `.xlsx` and `.csv` files
2. Loads files into in-memory DuckDB for preprocessing
3. Joins and validates data
4. Ingests into Neo4j knowledge graph

## Sample Data

See the included sample files for reference:
- `projects_sample.csv`
- `teams_sample.csv`
- `managers_sample.csv`

## Notes

- File names are case-insensitive
- The system will attempt to auto-detect table purposes based on column names
- Multiple files can contain similar data types (e.g., multiple project files will be merged)
- UTF-8 encoding is recommended for all files
