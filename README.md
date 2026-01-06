# Data Platform for Small Businesses

A Python web application that helps small companies transform their Excel/CSV data into a structured PostgreSQL database, then automatically generates AI-powered KPI dashboards tailored to their business.

## Features

### 📊 Data Import & Transformation
- **Excel & CSV to Database Conversion**: Automatically converts files to PostgreSQL tables
  - Company name → Schema name (single schema per company)
  - Excel: `filename_sheetname` → Table name
  - CSV: `filename` → Table name
  - File contents → Table rows
- **Recursive Folder Scanning**: Drop files in a watch folder and process all `.xlsx`, `.xls`, `.xlsm`, and `.csv` files (including subdirectories)
- **Smart Append Logic**: Automatically appends to existing tables when names match
- **LLM Column Standardization**: Uses OpenAI to standardize column names across files for consistency (e.g., "ID_no" → "id", "Customer Name" → "customer_name")

### 🤖 AI-Powered KPI Analytics
- **KPI Analyst Agent**: Analyzes your database schema and company context to suggest relevant, actionable KPIs
  - Industry-aware suggestions (ecommerce, SaaS, finance, healthcare, manufacturing, services, education)
  - Uses company description to focus on metrics that matter to YOUR business
  - Explores data with tools: schema summary, sample data, column statistics
- **Data Engineer Agent**: Executes KPI queries safely and computes values
  - SQL query validation (only SELECT statements allowed)
  - Automatic query fixing with LLM when errors occur
  - Proper value formatting (percentages, currency, large numbers)
- **KPI Dashboard**: Visual dashboard displaying all computed KPIs with categories and importance levels

### 🔧 Developer Features
- **Comprehensive Logging**: All KPI analysis and computation logged to `logs/` directory
  - `kpi_analysis.log` - Full prompts, tool calls, and LLM responses
  - `kpi_computation.log` - Query execution details and results
  - Logs reset on each new dashboard generation
- **Web UI**: Modern interface for uploading files, configuring company info, and viewing KPI dashboards

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL database (running via Docker or locally)
- OpenAI API key (required for KPI generation, optional for column standardization)

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

### Step 1: Import Your Data

**Option A: Web Upload**
1. Enter your company name in the input field
2. Drag and drop Excel or CSV files onto the upload zone
3. Monitor processing status in real-time

**Option B: Watch Folder**
1. Place Excel/CSV files in the `watch_folder/` directory
2. Enter company name and click "Scan & Process"
3. All files (including in subdirectories) will be processed

### Step 2: Generate KPI Dashboard

1. Navigate to the **KPI Dashboard** section
2. Select your schema (company) from the dropdown
3. Fill in:
   - **Company Name**: Your business name
   - **Industry**: Select from dropdown (ecommerce, SaaS, finance, etc.)
   - **Description**: Describe what your company does - this helps the AI suggest relevant KPIs
4. Click **Generate KPIs**
5. View your personalized KPI dashboard with computed values

## How It Works

### Data Import Flow

| File Type | Schema Name | Table Name |
|-----------|-------------|------------|
| Excel (.xlsx, .xls, .xlsm) | Company name | `{filename}_{sheetname}` |
| CSV (.csv) | Company name | `{filename}` |

**Examples:**
- Company: "Acme Corp", File: `sales_report.xlsx`, Sheet: "Q1 Data" → Schema: `acme_corp`, Table: `sales_report_q1_data`
- Company: "Acme Corp", File: `customers.csv` → Schema: `acme_corp`, Table: `customers`

### KPI Generation Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Company Info   │ ──▶ │  KPI Analyst     │ ──▶ │  Data Engineer  │
│  + Schema Data  │     │  (suggests KPIs) │     │  (computes KPIs)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  KPI Dashboard  │
                                                 │  (displays KPIs)│
                                                 └─────────────────┘
```

1. **KPI Analyst Agent** receives company info and explores the database using tools
2. Based on industry context and company description, suggests 5-10 relevant KPIs
3. **Data Engineer Agent** validates and executes each SQL query
4. Results are formatted and displayed in the dashboard

### Conflict Resolution (Data Import)

| Scenario | Action |
|----------|--------|
| New schema | Create schema |
| Existing schema | Use existing |
| New table | Create table |
| Existing table, same columns | Append rows |
| Existing table, new columns | ALTER TABLE ADD COLUMN |
| Existing table, missing columns | Insert NULL for missing |

## API Endpoints

### Data Import
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

### KPI Analytics
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kpi/analyze` | POST | Analyze schema and suggest KPIs |
| `/api/kpi/suggest` | POST | Get basic KPI suggestions (no LLM) |
| `/api/kpi/compute` | POST | Compute values for given KPIs |
| `/api/kpi/dashboard/{schema}` | GET | Full dashboard: analyze + compute |
| `/api/kpi/schema-summary/{schema}` | GET | Get schema summary for preview |

## Project Structure

```
DE_App/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration settings
│   ├── services/
│   │   ├── file_processor.py      # Excel & CSV parsing logic
│   │   ├── db_manager.py          # PostgreSQL operations + KPI tools
│   │   ├── llm_standardizer.py    # OpenAI column mapping
│   │   ├── kpi_analyst.py         # KPI Analyst Agent
│   │   ├── data_engineer.py       # Data Engineer Agent
│   │   └── agent_tools.py         # Shared agent tools
│   └── models/
│       ├── schemas.py             # Data import Pydantic models
│       └── kpi_schemas.py         # KPI Pydantic models
├── frontend/
│   └── index.html                 # Web UI
├── logs/                          # KPI analysis logs (gitignored)
│   ├── kpi_analysis.log
│   └── kpi_computation.log
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

# OpenAI Configuration (required for KPI generation)
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

## Debugging KPIs

If KPIs don't make sense or queries fail:

1. Check the logs in `logs/kpi_analysis.log` and `logs/kpi_computation.log`
2. Logs show:
   - Full system and user prompts sent to the LLM
   - Tool calls made by the agent (schema exploration)
   - Raw LLM responses
   - SQL queries executed
   - Query results and any errors
3. Logs are reset each time you generate a new dashboard

## License

MIT
