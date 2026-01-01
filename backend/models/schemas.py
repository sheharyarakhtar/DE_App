"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of file processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileInfo(BaseModel):
    """Information about a discovered Excel file."""
    file_path: str
    file_name: str
    file_size: int
    sheets: list[str] = []


class SheetData(BaseModel):
    """Data extracted from an Excel sheet."""
    sheet_name: str
    table_name: str
    columns: list[str]
    column_types: dict[str, str]
    row_count: int
    sample_data: list[dict[str, Any]] = []


class ExcelFileData(BaseModel):
    """Complete data from an Excel file."""
    file_path: str
    file_name: str
    schema_name: str
    sheets: list[SheetData]


class ColumnMapping(BaseModel):
    """Mapping from original column name to standardized name."""
    original_name: str
    standardized_name: str
    confidence: float = 1.0
    reason: Optional[str] = None


class StandardizationResult(BaseModel):
    """Result of LLM column standardization."""
    mappings: list[ColumnMapping]
    schema_name: str
    table_name: str


class ProcessingResult(BaseModel):
    """Result of processing a single file."""
    file_path: str
    schema_name: str
    tables_created: list[str]
    tables_updated: list[str]
    total_rows_inserted: int
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class ImportHistoryEntry(BaseModel):
    """Entry in the import history."""
    id: int
    file_name: str
    schema_name: str
    table_name: str
    rows_inserted: int
    columns_standardized: int
    processed_at: datetime
    status: ProcessingStatus


class UploadResponse(BaseModel):
    """Response for file upload endpoint."""
    message: str
    file_name: str
    status: ProcessingStatus
    result: Optional[ProcessingResult] = None


class ScanFolderRequest(BaseModel):
    """Request to scan a folder for Excel files."""
    folder_path: Optional[str] = None
    recursive: bool = True
    dry_run: bool = False


class ScanFolderResponse(BaseModel):
    """Response for folder scan endpoint."""
    message: str
    files_found: int
    files_processed: int
    results: list[ProcessingResult]


class StatusResponse(BaseModel):
    """Response for status endpoint."""
    is_processing: bool
    current_file: Optional[str] = None
    files_in_queue: int
    last_processed: Optional[str] = None


class SchemaInfo(BaseModel):
    """Information about a database schema."""
    schema_name: str
    tables: list[str]
    total_rows: int


class TableInfo(BaseModel):
    """Information about a database table."""
    schema_name: str
    table_name: str
    columns: list[dict[str, str]]
    row_count: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None

