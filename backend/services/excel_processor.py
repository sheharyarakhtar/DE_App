"""Excel file processor for parsing and extracting data."""

import os
import logging
from pathlib import Path
from typing import Any, Generator
from datetime import datetime

import pandas as pd
import openpyxl
import xlrd

from backend.services.db_manager import DatabaseManager
from backend.models.schemas import FileInfo, SheetData, ExcelFileData

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Processes Excel files and extracts data for database import."""
    
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls', '.xlsm'}
    
    def __init__(self):
        """Initialize the Excel processor."""
        self.db_manager = DatabaseManager()
    
    def discover_excel_files(
        self,
        folder_path: str,
        recursive: bool = True
    ) -> Generator[FileInfo, None, None]:
        """
        Discover all Excel files in a folder.
        
        Args:
            folder_path: Path to the folder to scan
            recursive: Whether to scan subdirectories
            
        Yields:
            FileInfo objects for each discovered Excel file
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return
        
        if not folder.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            return
        
        # Choose iteration method based on recursive flag
        if recursive:
            file_iterator = folder.rglob('*')
        else:
            file_iterator = folder.glob('*')
        
        for file_path in file_iterator:
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    # Get basic file info
                    file_size = file_path.stat().st_size
                    
                    # Try to get sheet names
                    sheets = self._get_sheet_names(str(file_path))
                    
                    yield FileInfo(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_size=file_size,
                        sheets=sheets
                    )
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
    
    def _get_sheet_names(self, file_path: str) -> list[str]:
        """Get sheet names from an Excel file."""
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.xls':
                workbook = xlrd.open_workbook(file_path, on_demand=True)
                return workbook.sheet_names()
            else:
                workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                return workbook.sheetnames
        except Exception as e:
            logger.error(f"Error reading sheet names from {file_path}: {e}")
            return []
    
    def parse_excel_file(self, file_path: str) -> ExcelFileData:
        """
        Parse an Excel file and extract all sheet data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            ExcelFileData containing all parsed information
        """
        path = Path(file_path)
        file_name = path.stem  # Filename without extension
        schema_name = DatabaseManager.sanitize_name(file_name)
        
        sheets_data = []
        sheet_names = self._get_sheet_names(file_path)
        
        for sheet_name in sheet_names:
            try:
                sheet_data = self._parse_sheet(file_path, sheet_name)
                if sheet_data:
                    sheets_data.append(sheet_data)
            except Exception as e:
                logger.error(f"Error parsing sheet {sheet_name} in {file_path}: {e}")
                continue
        
        return ExcelFileData(
            file_path=file_path,
            file_name=path.name,
            schema_name=schema_name,
            sheets=sheets_data
        )
    
    def _parse_sheet(self, file_path: str, sheet_name: str) -> SheetData | None:
        """
        Parse a single sheet from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to parse
            
        Returns:
            SheetData or None if sheet is empty
        """
        try:
            # Read the sheet using pandas
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine=None)
            
            # Skip empty sheets
            if df.empty or len(df.columns) == 0:
                logger.info(f"Skipping empty sheet: {sheet_name}")
                return None
            
            # Clean up column names
            df.columns = [str(col).strip() if pd.notna(col) else f'column_{i}' 
                         for i, col in enumerate(df.columns)]
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            if df.empty:
                logger.info(f"Sheet {sheet_name} has no data after removing empty rows")
                return None
            
            # Infer column types
            column_types = {}
            for col in df.columns:
                sample_values = df[col].dropna().head(100).tolist()
                column_types[col] = self.db_manager.infer_postgres_type(sample_values)
            
            # Get sample data (first 5 rows)
            sample_data = df.head(5).to_dict(orient='records')
            
            # Clean sample data (convert NaN to None)
            for row in sample_data:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
            
            table_name = DatabaseManager.sanitize_name(sheet_name)
            
            return SheetData(
                sheet_name=sheet_name,
                table_name=table_name,
                columns=list(df.columns),
                column_types=column_types,
                row_count=len(df),
                sample_data=sample_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing sheet {sheet_name}: {e}")
            raise
    
    def get_sheet_dataframe(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Get a pandas DataFrame for a specific sheet.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet
            
        Returns:
            pandas DataFrame with the sheet data
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine=None)
        
        # Clean up column names
        df.columns = [str(col).strip() if pd.notna(col) else f'column_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        return df
    
    def prepare_data_for_insert(
        self,
        df: pd.DataFrame,
        column_mapping: dict[str, str] | None = None
    ) -> tuple[list[str], list[list[Any]]]:
        """
        Prepare DataFrame data for database insertion.
        
        Args:
            df: pandas DataFrame with the data
            column_mapping: Optional mapping from original to standardized column names
            
        Returns:
            Tuple of (column_names, data_rows)
        """
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        columns = list(df.columns)
        
        # Convert DataFrame to list of lists, handling special values
        data = []
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                # Handle NaN/NaT values
                if pd.isna(val):
                    row_data.append(None)
                # Handle datetime objects
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_data.append(val.isoformat() if pd.notna(val) else None)
                # Handle other types
                else:
                    row_data.append(val)
            data.append(row_data)
        
        return columns, data
    
    def process_file(
        self,
        file_path: str,
        column_mappings: dict[str, dict[str, str]] | None = None,
        dry_run: bool = False
    ) -> dict:
        """
        Process an Excel file and import it to the database.
        
        Args:
            file_path: Path to the Excel file
            column_mappings: Optional dict mapping table names to column mappings
            dry_run: If True, don't actually insert data
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        results = {
            'file_path': file_path,
            'schema_name': '',
            'tables_created': [],
            'tables_updated': [],
            'total_rows_inserted': 0,
            'errors': []
        }
        
        try:
            # Parse the Excel file
            excel_data = self.parse_excel_file(file_path)
            results['schema_name'] = excel_data.schema_name
            
            if dry_run:
                results['dry_run'] = True
                results['sheets'] = [
                    {
                        'table_name': sheet.table_name,
                        'columns': sheet.columns,
                        'row_count': sheet.row_count
                    }
                    for sheet in excel_data.sheets
                ]
                return results
            
            # Create schema if needed
            self.db_manager.create_schema(excel_data.schema_name)
            
            # Process each sheet
            for sheet in excel_data.sheets:
                try:
                    # Get column mapping for this table if provided
                    table_column_mapping = None
                    if column_mappings and sheet.table_name in column_mappings:
                        table_column_mapping = column_mappings[sheet.table_name]
                    
                    # Get the full DataFrame
                    df = self.get_sheet_dataframe(file_path, sheet.sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Prepare data for insertion
                    columns, data = self.prepare_data_for_insert(df, table_column_mapping)
                    
                    # Determine column types
                    column_types = {}
                    for col in columns:
                        original_col = col
                        if table_column_mapping:
                            # Find original column name
                            for orig, mapped in table_column_mapping.items():
                                if mapped == col:
                                    original_col = orig
                                    break
                        
                        if original_col in sheet.column_types:
                            column_types[col] = sheet.column_types[original_col]
                        else:
                            column_types[col] = 'TEXT'
                    
                    # Check if table exists
                    table_exists = self.db_manager.table_exists(
                        excel_data.schema_name,
                        sheet.table_name
                    )
                    
                    if table_exists:
                        # Add any new columns
                        self.db_manager.add_columns(
                            excel_data.schema_name,
                            sheet.table_name,
                            column_types
                        )
                        results['tables_updated'].append(sheet.table_name)
                    else:
                        # Create new table
                        self.db_manager.create_table(
                            excel_data.schema_name,
                            sheet.table_name,
                            column_types
                        )
                        results['tables_created'].append(sheet.table_name)
                    
                    # Insert data
                    rows_inserted = self.db_manager.insert_data(
                        excel_data.schema_name,
                        sheet.table_name,
                        columns,
                        data
                    )
                    results['total_rows_inserted'] += rows_inserted
                    
                    # Log the import
                    self.db_manager.log_import(
                        file_name=excel_data.file_name,
                        file_path=file_path,
                        schema_name=excel_data.schema_name,
                        table_name=sheet.table_name,
                        rows_inserted=rows_inserted,
                        columns_standardized=len(table_column_mapping) if table_column_mapping else 0,
                        column_mappings=table_column_mapping,
                        status='completed'
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing sheet {sheet.sheet_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    
                    # Rollback the failed transaction so we can continue with other sheets
                    self.db_manager.rollback()
                    
                    # Log the failed import (after rollback so this succeeds)
                    try:
                        self.db_manager.log_import(
                            file_name=excel_data.file_name,
                            file_path=file_path,
                            schema_name=excel_data.schema_name,
                            table_name=sheet.table_name,
                            rows_inserted=0,
                            columns_standardized=0,
                            column_mappings=None,
                            status='failed',
                            error_message=str(e)
                        )
                    except Exception as log_error:
                        logger.error(f"Failed to log import error: {log_error}")
            
            results['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
            return results
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            results['errors'].append(str(e))
            results['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
            return results
        
        finally:
            self.db_manager.close()

