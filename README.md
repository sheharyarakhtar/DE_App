# Data-to-PostgreSQL Converter

A Python web application that converts Excel and CSV files into structured PostgreSQL database tables with intelligent column standardization powered by OpenAI.

## Features

- **Excel & CSV to Database Conversion**: Automatically converts files to PostgreSQL tables
  - Company name → Schema name (single schema per company)
  - Excel: `filename_sheetname` → Table name
  - CSV: `filename` → Table name
  - File contents → Table rows
- **Recursive Folder Scanning**: Drop files in a watch folder and process all `.xlsx`, `.xls`, `.xlsm`, and `.csv` files (including subdirectories)
- **Smart Append Logic**: Automatically appends to existing tables when names match
- **LLM Column Standardization**: Uses OpenAI to standardize column names across files for consistency (e.g., "ID_no" → "id", "Customer Name" → "customer_name")
- **Web UI**: Simple interface for uploading files, entering company name, and monitoring processing status

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL database (running via Docker or locally)
- OpenAI API key (optional, for column standardization)

### Installation

1. Clone and navigate to the project:
   ```bash
   cd DE_App
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and OpenAI API key
   ```

5. Start the application:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. Open http://localhost:8000 in your browser

## Usage

### Step 1: Enter Company Name
Enter your company/organization name in the input field at the top. This becomes the PostgreSQL schema name where all your data will be stored.

### Step 2: Upload Files

**Option A: Web Upload**
1. Drag and drop an Excel or CSV file onto the upload zone
2. Or click to browse and select a file
3. Monitor processing status in real-time

**Option B: Watch Folder**
1. Place Excel/CSV files in the `watch_folder/` directory
2. Click "Scan & Process" in the web UI
3. All files (including in subdirectories) will be processed

## How It Works

### Schema & Table Naming

| File Type | Schema Name | Table Name |
|-----------|-------------|------------|
| Excel (.xlsx, .xls, .xlsm) | Company name | `{filename}_{sheetname}` |
| CSV (.csv) | Company name | `{filename}` |

**Examples:**
- Company: "Acme Corp", File: `sales_report.xlsx`, Sheet: "Q1 Data" → Schema: `acme_corp`, Table: `sales_report_q1_data`
- Company: "Acme Corp", File: `customers.csv` → Schema: `acme_corp`, Table: `customers`

### Conflict Resolution

| Scenario | Action |
|----------|--------|
| New schema | Create schema |
| Existing schema | Use existing |
| New table | Create table |
| Existing table, same columns | Append rows |
| Existing table, new columns | ALTER TABLE ADD COLUMN |
| Existing table, missing columns | Insert NULL for missing |

### LLM Column Standardization

When enabled, the application uses OpenAI to standardize column names:
- Analyzes incoming file columns against existing database columns
- Suggests mappings (e.g., "ID_no" → "id", "Customer Name" → "customer_name")
- Ensures data consistency across multiple file imports
- Caches mappings to reduce API calls

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload` | POST | Upload single file (requires `company_name` form field) |
| `/api/scan-folder` | POST | Scan watch folder (requires `company_name` in body) |
| `/api/status` | GET | Get processing status |
| `/api/history` | GET | Get import history |
| `/api/schemas` | GET | List all schemas in database |
| `/api/schemas/{name}/tables` | GET | List tables in a schema |
| `/api/watch-folder` | GET | Get watch folder contents |
| `/api/health` | GET | Health check |

## Project Structure

```
DE_App/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration settings
│   ├── services/
│   │   ├── file_processor.py      # Excel & CSV parsing logic
│   │   ├── db_manager.py          # PostgreSQL operations
│   │   └── llm_standardizer.py    # OpenAI column mapping
│   └── models/
│       └── schemas.py             # Pydantic models
├── frontend/
│   └── index.html                 # Web UI
├── watch_folder/                  # Drop files here for batch processing
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuration

Environment variables in `.env`:

```bash
# PostgreSQL Configuration
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppassword
POSTGRES_DB=appdb

# OpenAI Configuration (optional)
OPENAI_API_KEY=sk-your-key-here

# Application Configuration
WATCH_FOLDER=./watch_folder
```

## Docker PostgreSQL Setup

If you need to set up PostgreSQL via Docker:

```bash
# Create network
docker network create pg-net

# Run PostgreSQL
docker run -d \
  --name postgres \
  --network pg-net \
  -e POSTGRES_USER=appuser \
  -e POSTGRES_PASSWORD=apppassword \
  -e POSTGRES_DB=appdb \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16

# Optional: Run pgAdmin for database management
docker run -d \
  --name pgadmin \
  --network pg-net \
  -e PGADMIN_DEFAULT_EMAIL=admin@example.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin \
  -p 5050:80 \
  dpage/pgadmin4
```

Access pgAdmin at http://localhost:5050

## Supported File Types

- **Excel**: `.xlsx`, `.xls`, `.xlsm`
- **CSV**: `.csv` (auto-detects encoding: UTF-8, Latin-1, CP1252)

## License

MIT
