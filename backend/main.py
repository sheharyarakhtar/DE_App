"""FastAPI application for Excel/CSV-to-PostgreSQL converter."""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings, get_runtime_config, DBConfigRequest, OpenAIConfigRequest, ConfigStatusResponse
from backend.services.file_processor import FileProcessor
from backend.services.db_manager import DatabaseManager
from backend.services.llm_standardizer import LLMStandardizer
from backend.models.schemas import (
    UploadResponse,
    ScanFolderRequest,
    ScanFolderResponse,
    StatusResponse,
    ProcessingResult,
    ProcessingStatus,
    SchemaInfo,
    TableInfo,
    ErrorResponse
)
from backend.models.kpi_schemas import (
    KPIAnalysisRequest,
    KPIAnalysisResponse,
    KPIComputeRequest,
    KPIComputeResponse,
    KPIDefinition,
    KPIDashboard
)
from backend.services.kpi_analyst import KPIAnalyst, reset_kpi_logs
from backend.services.data_engineer import DataEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state
class AppState:
    is_processing: bool = False
    current_file: Optional[str] = None
    files_in_queue: int = 0
    last_processed: Optional[str] = None
    processing_results: list[ProcessingResult] = []

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    
    # Ensure watch folder exists
    watch_folder = Path(settings.watch_folder)
    watch_folder.mkdir(parents=True, exist_ok=True)
    
    # Ensure temp upload folder exists
    temp_folder = Path("temp_uploads")
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Watch folder: {watch_folder.absolute()}")
    logger.info("Application started")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title="Data-to-PostgreSQL Converter",
    description="Convert Excel and CSV files to structured PostgreSQL database tables with LLM-powered column standardization",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_single_file(
    file_path: str,
    company_name: str,
    use_llm: bool = True
) -> ProcessingResult:
    """
    Process a single file (Excel or CSV).
    
    Args:
        file_path: Path to the file
        company_name: Company name to use as schema
        use_llm: Whether to use LLM for column standardization
        
    Returns:
        ProcessingResult with processing details
    """
    start_time = datetime.now()
    processor = FileProcessor()
    standardizer = LLMStandardizer() if use_llm else None
    
    try:
        # Parse the file first to get column info
        file_data = processor.parse_file(file_path, company_name)
        
        # Get column mappings from LLM if enabled
        column_mappings = {}
        if standardizer and standardizer.client:
            for sheet in file_data.sheets:
                try:
                    result = standardizer.standardize_columns(
                        file_data.schema_name,
                        sheet.table_name,
                        sheet.columns
                    )
                    column_mappings[sheet.table_name] = standardizer.get_mapping_dict(result)
                    logger.info(f"Standardized columns for {sheet.table_name}: {column_mappings[sheet.table_name]}")
                except Exception as e:
                    logger.error(f"Error standardizing columns for {sheet.table_name}: {e}")
        
        # Process the file with column mappings
        result = processor.process_file(file_path, company_name, column_mappings)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            file_path=file_path,
            schema_name=result['schema_name'],
            tables_created=result['tables_created'],
            tables_updated=result['tables_updated'],
            total_rows_inserted=result['total_rows_inserted'],
            status=ProcessingStatus.COMPLETED if not result['errors'] else ProcessingStatus.FAILED,
            error_message='; '.join(result['errors']) if result['errors'] else None,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return ProcessingResult(
            file_path=file_path,
            schema_name=DatabaseManager.sanitize_name(company_name),
            tables_created=[],
            tables_updated=[],
            total_rows_inserted=0,
            status=ProcessingStatus.FAILED,
            error_message=str(e),
            processing_time_seconds=processing_time
        )
    finally:
        if standardizer:
            standardizer.close()


def process_files_background(file_paths: list[str], company_name: str, use_llm: bool = True):
    """Background task to process multiple files."""
    global app_state
    
    app_state.is_processing = True
    app_state.files_in_queue = len(file_paths)
    app_state.processing_results = []
    
    for file_path in file_paths:
        app_state.current_file = file_path
        
        result = process_single_file(file_path, company_name, use_llm)
        app_state.processing_results.append(result)
        
        app_state.files_in_queue -= 1
        app_state.last_processed = file_path
    
    app_state.is_processing = False
    app_state.current_file = None


# Serve frontend
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_name: str = Form(...),
    use_llm: bool = Form(default=True)
):
    """
    Upload and process a single file (Excel or CSV).
    
    The file will be processed and imported into the database.
    Schema will be named after the company.
    Table names will be filename_sheetname for Excel, or filename for CSV.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not company_name or not company_name.strip():
        raise HTTPException(status_code=400, detail="Company name is required")
    
    ext = Path(file.filename).suffix.lower()
    supported_extensions = {'.xlsx', '.xls', '.xlsm', '.csv'}
    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .xlsx, .xls, .xlsm, .csv"
        )
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / file.filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        result = process_single_file(str(temp_path), company_name.strip(), use_llm)
        
        return UploadResponse(
            message=f"File processed successfully" if result.status == ProcessingStatus.COMPLETED else "File processing failed",
            file_name=file.filename,
            status=result.status,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/scan-folder", response_model=ScanFolderResponse)
async def scan_folder(
    background_tasks: BackgroundTasks,
    request: ScanFolderRequest = None
):
    """
    Scan a folder for Excel and CSV files and process them.
    
    If no folder path is provided, uses the configured watch folder.
    Company name is required for processing.
    """
    settings = get_settings()
    
    # Use provided folder or default watch folder
    folder_path = request.folder_path if request and request.folder_path else settings.watch_folder
    recursive = request.recursive if request else True
    dry_run = request.dry_run if request else False
    company_name = request.company_name if request and request.company_name else None
    
    if not dry_run and not company_name:
        raise HTTPException(status_code=400, detail="Company name is required for processing")
    
    # Validate folder exists
    folder = Path(folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")
    
    # Discover files (Excel and CSV)
    processor = FileProcessor()
    files = list(processor.discover_files(str(folder), recursive))
    
    if dry_run:
        return ScanFolderResponse(
            message=f"Found {len(files)} file(s) (dry run)",
            files_found=len(files),
            files_processed=0,
            results=[
                ProcessingResult(
                    file_path=f.file_path,
                    schema_name=DatabaseManager.sanitize_name(company_name) if company_name else "preview",
                    tables_created=[],
                    tables_updated=[],
                    total_rows_inserted=0,
                    status=ProcessingStatus.PENDING
                )
                for f in files
            ]
        )
    
    if not files:
        return ScanFolderResponse(
            message="No supported files found in the specified folder",
            files_found=0,
            files_processed=0,
            results=[]
        )
    
    # Process files
    results = []
    for file_info in files:
        result = process_single_file(file_info.file_path, company_name)
        results.append(result)
    
    successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
    
    return ScanFolderResponse(
        message=f"Processed {successful}/{len(files)} files successfully",
        files_found=len(files),
        files_processed=successful,
        results=results
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get the current processing status."""
    return StatusResponse(
        is_processing=app_state.is_processing,
        current_file=app_state.current_file,
        files_in_queue=app_state.files_in_queue,
        last_processed=app_state.last_processed
    )


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get import history."""
    db_manager = DatabaseManager()
    try:
        history = db_manager.get_import_history(limit)
        # Convert datetime objects to strings for JSON serialization
        for entry in history:
            if 'processed_at' in entry and entry['processed_at']:
                entry['processed_at'] = entry['processed_at'].isoformat()
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_manager.close()


@app.get("/api/schemas")
async def list_schemas():
    """List all schemas in the database."""
    db_manager = DatabaseManager()
    try:
        schemas = db_manager.get_all_schemas()
        
        schema_info = []
        for schema in schemas:
            tables = db_manager.get_schema_tables(schema)
            total_rows = 0
            for table in tables:
                try:
                    total_rows += db_manager.get_table_row_count(schema, table)
                except:
                    pass
            
            schema_info.append(SchemaInfo(
                schema_name=schema,
                tables=tables,
                total_rows=total_rows
            ))
        
        return {"schemas": [s.model_dump() for s in schema_info]}
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_manager.close()


@app.get("/api/schemas/{schema_name}/tables")
async def list_tables(schema_name: str):
    """List all tables in a schema."""
    db_manager = DatabaseManager()
    try:
        tables = db_manager.get_schema_tables(schema_name)
        
        table_info = []
        for table in tables:
            columns = db_manager.get_table_columns(schema_name, table)
            row_count = db_manager.get_table_row_count(schema_name, table)
            
            table_info.append(TableInfo(
                schema_name=schema_name,
                table_name=table,
                columns=columns,
                row_count=row_count
            ))
        
        return {"tables": [t.model_dump() for t in table_info]}
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_manager.close()


@app.get("/api/watch-folder")
async def get_watch_folder():
    """Get the configured watch folder path and its contents."""
    settings = get_settings()
    folder = Path(settings.watch_folder)
    
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    processor = FileProcessor()
    files = list(processor.discover_files(str(folder), recursive=True))
    
    return {
        "path": str(folder.absolute()),
        "files": [
            {
                "file_path": f.file_path,
                "file_name": f.file_name,
                "file_size": f.file_size,
                "sheets": f.sheets
            }
            for f in files
        ]
    }


@app.delete("/api/cache")
async def clear_cache():
    """Clear the LLM column mapping cache."""
    standardizer = LLMStandardizer()
    standardizer.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    db_manager = DatabaseManager()
    try:
        # Try to connect to the database
        conn = db_manager._get_connection()
        db_status = "connected" if conn and not conn.closed else "disconnected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    finally:
        db_manager.close()
    
    settings = get_settings()
    
    return {
        "status": "healthy",
        "database": db_status,
        "openai_configured": bool(settings.openai_api_key),
        "watch_folder": settings.watch_folder
    }


# ==================== KPI Endpoints ====================

@app.post("/api/kpi/analyze", response_model=KPIAnalysisResponse)
async def analyze_kpis(request: KPIAnalysisRequest):
    """
    Analyze a schema and suggest relevant KPIs based on the data and company context.
    
    Uses AI to understand the data structure and suggest meaningful KPIs.
    """
    analyst = KPIAnalyst()
    try:
        if not analyst.client:
            # Fall back to basic KPIs if no OpenAI key
            basic_kpis = analyst.suggest_basic_kpis(request.schema_name)
            return KPIAnalysisResponse(
                schema_name=request.schema_name,
                company_name=request.company_name,
                kpis=basic_kpis,
                analysis_summary="Generated basic KPIs (AI not configured). Add OpenAI API key for intelligent KPI suggestions."
            )
        
        return analyst.analyze(request)
    except Exception as e:
        logger.error(f"Error analyzing KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        analyst.close()


@app.post("/api/kpi/suggest")
async def suggest_kpis(
    schema_name: str,
    company_name: str,
    company_description: Optional[str] = None,
    industry: Optional[str] = None,
    max_kpis: int = 10
):
    """
    Get KPI suggestions without computing them.
    
    Lighter weight endpoint that just returns suggestions.
    """
    request = KPIAnalysisRequest(
        schema_name=schema_name,
        company_name=company_name,
        company_description=company_description,
        industry=industry,
        max_kpis=max_kpis
    )
    return await analyze_kpis(request)


@app.post("/api/kpi/compute", response_model=KPIComputeResponse)
async def compute_kpis(request: KPIComputeRequest):
    """
    Compute specific KPIs.
    
    Takes KPI definitions and executes their SQL queries to get current values.
    """
    engineer = DataEngineer()
    try:
        return engineer.compute_kpis_with_retry(request)
    except Exception as e:
        logger.error(f"Error computing KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engineer.close()


@app.get("/api/kpi/dashboard/{schema_name}")
async def get_kpi_dashboard(
    schema_name: str,
    company_name: str,
    company_description: Optional[str] = None,
    industry: Optional[str] = None,
    table_name: Optional[str] = None
):
    """
    Get a complete KPI dashboard for a schema.
    
    This is a convenience endpoint that:
    1. Analyzes the schema and suggests KPIs
    2. Computes all suggested KPIs
    3. Returns the complete dashboard
    
    Args:
        schema_name: Database schema to analyze
        company_name: Name of the company
        company_description: Optional description of the company
        industry: Optional industry/vertical
        table_name: Optional specific table to analyze (if None, analyze all tables)
    """
    # Reset logs for fresh dashboard generation
    reset_kpi_logs()
    
    analyst = KPIAnalyst()
    engineer = DataEngineer()
    
    try:
        # Step 1: Analyze and get KPI suggestions
        analysis_request = KPIAnalysisRequest(
            schema_name=schema_name,
            table_name=table_name,
            company_name=company_name,
            company_description=company_description,
            industry=industry,
            max_kpis=10
        )
        
        if analyst.client:
            analysis = analyst.analyze(analysis_request)
        else:
            basic_kpis = analyst.suggest_basic_kpis(schema_name, table_name)
            analysis = KPIAnalysisResponse(
                schema_name=schema_name,
                company_name=company_name,
                kpis=basic_kpis,
                analysis_summary="Generated basic KPIs (AI not configured)"
            )
        
        # Step 2: Compute the KPIs
        compute_request = KPIComputeRequest(
            schema_name=schema_name,
            kpis=analysis.kpis
        )
        
        computed = engineer.compute_kpis_with_retry(compute_request)
        
        # Step 3: Build dashboard response
        return KPIDashboard(
            schema_name=schema_name,
            table_name=table_name,
            company_name=company_name,
            company_description=company_description,
            industry=industry,
            kpis=computed.results,
            generated_at=analysis.generated_at,
            last_computed_at=computed.computed_at
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error getting KPI dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        analyst.close()
        engineer.close()


@app.get("/api/kpi/schema-summary/{schema_name}")
async def get_schema_summary(schema_name: str):
    """
    Get a summary of a schema for KPI analysis preview.
    
    Returns tables, columns, and row counts.
    """
    db_manager = DatabaseManager()
    try:
        summary = db_manager.get_schema_summary(schema_name)
        return summary
    except Exception as e:
        logger.error(f"Error getting schema summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_manager.close()


# ==================== Configuration Endpoints ====================

@app.post("/api/config/test-db")
async def test_db_connection(config: DBConfigRequest):
    """
    Test database connection with provided credentials.
    Does not save the configuration.
    """
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            connect_timeout=5
        )
        conn.close()
        return {"success": True, "message": "Connection successful"}
    except psycopg2.OperationalError as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/api/config/test-openai")
async def test_openai_key(config: OpenAIConfigRequest):
    """
    Test OpenAI API key validity.
    Does not save the configuration.
    """
    from openai import OpenAI
    
    if not config.api_key or not config.api_key.startswith("sk-"):
        return {"success": False, "message": "Invalid API key format"}
    
    try:
        client = OpenAI(api_key=config.api_key)
        # Make a minimal API call to verify the key
        client.models.list()
        return {"success": True, "message": "API key is valid"}
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            return {"success": False, "message": "Invalid API key"}
        elif "quota" in error_msg.lower():
            return {"success": True, "message": "API key valid (quota may be limited)"}
        else:
            return {"success": False, "message": f"Error: {error_msg[:100]}"}


@app.post("/api/config/set-db")
async def set_db_config(config: DBConfigRequest):
    """
    Set database configuration for the current session.
    Configuration is stored in memory and must be re-sent on server restart.
    """
    runtime_config = get_runtime_config()
    runtime_config.set_db_config(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database
    )
    logger.info(f"Database configuration updated: {config.host}:{config.port}/{config.database}")
    return {"success": True, "message": "Database configuration saved"}


@app.post("/api/config/set-openai")
async def set_openai_config(config: OpenAIConfigRequest):
    """
    Set OpenAI API key for the current session.
    Configuration is stored in memory and must be re-sent on server restart.
    """
    runtime_config = get_runtime_config()
    runtime_config.set_openai_key(config.api_key)
    logger.info("OpenAI API key updated")
    return {"success": True, "message": "OpenAI API key saved"}


@app.get("/api/config/status", response_model=ConfigStatusResponse)
async def get_config_status():
    """
    Get current configuration status.
    Returns whether DB and OpenAI are configured via UI (runtime) only.
    .env values are treated as fallback, not as "configured".
    """
    settings = get_settings()
    runtime_config = get_runtime_config()
    
    # Only consider "configured" if user explicitly set via UI
    db_config = runtime_config.db_config
    db_configured = db_config is not None
    db_source = "runtime" if db_configured else "none"
    
    # Only test DB connection if user has configured credentials via UI
    db_connected = False
    if db_configured and db_config:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["database"],
                connect_timeout=3
            )
            conn.close()
            db_connected = True
        except:
            pass
    
    # Only consider OpenAI "configured" if user explicitly set via UI
    openai_configured = runtime_config.openai_key is not None
    openai_source = "runtime" if openai_configured else "none"
    
    return ConfigStatusResponse(
        db_configured=db_configured,
        db_connected=db_connected,
        db_source=db_source,
        openai_configured=openai_configured,
        openai_source=openai_source
    )


@app.delete("/api/config/clear-db")
async def clear_db_config():
    """Clear runtime database configuration, reverting to .env or defaults."""
    runtime_config = get_runtime_config()
    runtime_config.clear_db_config()
    logger.info("Runtime database configuration cleared")
    return {"success": True, "message": "Database configuration cleared"}


@app.delete("/api/config/clear-openai")
async def clear_openai_config():
    """Clear runtime OpenAI configuration, reverting to .env or defaults."""
    runtime_config = get_runtime_config()
    runtime_config.clear_openai_key()
    logger.info("Runtime OpenAI configuration cleared")
    return {"success": True, "message": "OpenAI configuration cleared"}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
