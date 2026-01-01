# Excel-to-PostgreSQL Database Converter

A Python web application that converts Excel files into structured PostgreSQL database tables with intelligent column standardization powered by OpenAI.

## Features

- **Excel to Database Conversion**: Automatically converts Excel files to PostgreSQL schemas and tables
  - Filename → Schema name
  - Sheet name → Table name  
  - Sheet contents → Table rows
- **Recursive Folder Scanning**: Drop Excel files in a watch folder and process all `.xlsx` and `.xls` files
- **Smart Append Logic**: Automatically appends to existing schemas/tables when names match
- **LLM Column Standardization**: Uses OpenAI to standardize column names across files for consistency
- **Web UI**: Simple interface for uploading files and monitoring processing status

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL database (running via Docker or locally)
- OpenAI API key

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

### Option 1: Web Upload
1. Navigate to http://localhost:8000
2. Upload an Excel file using the web interface
3. Monitor processing status in real-time

### Option 2: Watch Folder
1. Place Excel files in the `watch_folder/` directory
2. Click "Scan Folder" in the web UI or call the API endpoint
3. All Excel files (including in subdirectories) will be processed

## How It Works

### Schema/Table Naming
- `sales_report.xlsx` → Schema: `sales_report`
- Sheet "Q1 Data" → Table: `q1_data`
- Special characters and spaces are converted to underscores

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
The application uses OpenAI to standardize column names:
- Analyzes incoming Excel columns against existing database columns
- Suggests mappings (e.g., "ID_no" → "id", "Customer Name" → "customer_name")
- Ensures data consistency across multiple file imports

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload` | POST | Upload single Excel file |
| `/api/scan-folder` | POST | Scan watch folder for Excel files |
| `/api/status` | GET | Get processing status |
| `/api/history` | GET | Get import history |
| `/api/schemas` | GET | List all schemas in database |

## Project Structure

```
DE_App/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration settings
│   ├── services/
│   │   ├── excel_processor.py    # Excel parsing logic
│   │   ├── db_manager.py         # PostgreSQL operations
│   │   └── llm_standardizer.py   # OpenAI column mapping
│   └── models/
│       └── schemas.py       # Pydantic models
├── frontend/
│   └── index.html           # Web UI
├── watch_folder/            # Drop Excel files here
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Environment variables in `.env`:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppassword
POSTGRES_DB=appdb
OPENAI_API_KEY=sk-...
WATCH_FOLDER=./watch_folder
```

## License

MIT

