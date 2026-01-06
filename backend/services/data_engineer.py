"""Data Engineer Agent - Executes KPI queries and computes values safely."""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Any

from openai import OpenAI

from backend.config import get_settings
from backend.services.db_manager import DatabaseManager
from backend.services.agent_tools import AgentToolExecutor, QUERY_TOOLS
from backend.models.kpi_schemas import (
    KPIDefinition,
    KPIResult,
    KPIComputeRequest,
    KPIComputeResponse,
    KPICategory,
    KPIImportance
)

# Set up file logging for KPI operations
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

kpi_logger = logging.getLogger("data_engineer")
kpi_logger.setLevel(logging.DEBUG)

# File handler - logs to file
log_file = os.path.join(LOG_DIR, "kpi_computation.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
kpi_logger.addHandler(file_handler)

# Console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
kpi_logger.addHandler(console_handler)

logger = kpi_logger


def format_value(value: Any, format_type: str = "auto") -> str:
    """
    Format a value for display with proper rounding.
    
    Args:
        value: The value to format
        format_type: How to format (auto, currency, percentage, number, text)
        
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    # Handle string values that might be numeric
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value
    
    if format_type == "currency":
        if isinstance(value, (int, float)):
            if abs(value) >= 1000000:
                return f"${value/1000000:,.2f}M"
            elif abs(value) >= 1000:
                return f"${value/1000:,.2f}K"
            else:
                return f"${value:,.2f}"
        return str(value)
    
    elif format_type == "percentage":
        if isinstance(value, (int, float)):
            return f"{value:.1f}%"
        return str(value)
    
    elif format_type == "number":
        if isinstance(value, (int, float)):
            # Round to 2 decimal places for display
            if isinstance(value, float):
                # Check if it's essentially an integer
                if value == int(value):
                    value = int(value)
                else:
                    value = round(value, 2)
            
            if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
                # Format integers with commas
                if abs(value) >= 1000000:
                    return f"{value/1000000:,.2f}M"
                elif abs(value) >= 1000:
                    return f"{value/1000:,.1f}K"
                else:
                    return f"{int(value):,}"
            else:
                # Format floats with 2 decimal places
                if abs(value) >= 1000000:
                    return f"{value/1000000:,.2f}M"
                elif abs(value) >= 1000:
                    return f"{value/1000:,.2f}K"
                else:
                    return f"{value:,.2f}"
        return str(value)
    
    else:  # auto
        if isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, float):
            # Round floats to 2 decimal places
            if value == int(value):
                value = int(value)
            else:
                value = round(value, 2)
            
            if abs(value) >= 1000000:
                return f"{value/1000000:,.2f}M"
            elif abs(value) >= 1000:
                return f"{value/1000:,.2f}K"
            elif isinstance(value, int) or value == int(value):
                return f"{int(value):,}"
            else:
                return f"{value:,.2f}"
        elif isinstance(value, int):
            if abs(value) >= 1000000:
                return f"{value/1000000:,.2f}M"
            elif abs(value) >= 1000:
                return f"{value/1000:,.1f}K"
            else:
                return f"{value:,}"
        else:
            return str(value)


class DataEngineer:
    """Agent that executes KPI queries and computes values."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.tool_executor = AgentToolExecutor()
        
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
    
    def compute_kpis(self, request: KPIComputeRequest) -> KPIComputeResponse:
        """
        Compute all KPIs in the request.
        """
        logger.info("=" * 80)
        logger.info("DATA ENGINEER - Computing KPIs")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Schema: {request.schema_name}")
        logger.info(f"Number of KPIs to compute: {len(request.kpis)}")
        
        results = []
        success_count = 0
        error_count = 0
        
        for i, kpi in enumerate(request.kpis):
            logger.info(f"\n--- Computing KPI {i+1}/{len(request.kpis)}: {kpi.name} ---")
            result = self.compute_single_kpi(kpi, request.schema_name)
            results.append(result)
            
            if result.error:
                error_count += 1
            else:
                success_count += 1
        
        logger.info(f"\nComputation complete: {success_count} success, {error_count} errors")
        logger.info("=" * 80)
        
        return KPIComputeResponse(
            schema_name=request.schema_name,
            results=results,
            success_count=success_count,
            error_count=error_count
        )
    
    def compute_single_kpi(self, kpi: KPIDefinition, schema_name: str) -> KPIResult:
        """
        Compute a single KPI.
        """
        logger.info(f"  KPI: {kpi.name}")
        logger.info(f"  Format Type: {kpi.format_type}")
        logger.info(f"  SQL Query: {kpi.sql_query}")
        
        try:
            if not self._validate_query(kpi.sql_query):
                logger.error(f"  VALIDATION FAILED: Query is not a valid SELECT statement")
                return KPIResult(
                    name=kpi.name,
                    description=kpi.description,
                    value=None,
                    formatted_value="Error",
                    category=kpi.category,
                    importance=kpi.importance,
                    sql_query=kpi.sql_query,
                    error="Query validation failed: Only SELECT statements are allowed"
                )
            
            logger.info(f"  Validation: PASSED")
            logger.info(f"  Executing query...")
            
            result = self.db_manager.execute_read_query(kpi.sql_query, schema_name)
            
            logger.info(f"  Query Result - Columns: {result.get('columns', [])}")
            logger.info(f"  Query Result - Row Count: {result.get('row_count', 0)}")
            logger.info(f"  Query Result - Rows: {result.get('rows', [])}")
            
            value = None
            if result["rows"] and len(result["rows"]) > 0:
                first_row = result["rows"][0]
                if first_row:
                    value = list(first_row.values())[0]
                    logger.info(f"  Raw Value: {value} (type: {type(value).__name__})")
                    
                    # Round numeric values
                    if isinstance(value, float):
                        if value == int(value):
                            value = int(value)
                        else:
                            value = round(value, 2)
                        logger.info(f"  Rounded Value: {value}")
            else:
                logger.warning(f"  No rows returned from query!")
            
            # Format the value using our improved formatter
            formatted_value = format_value(value, kpi.format_type)
            logger.info(f"  Formatted Value: {formatted_value}")
            
            # Add unit if specified
            if kpi.unit and value is not None:
                if kpi.format_type not in ("currency", "percentage"):
                    formatted_value = f"{formatted_value} {kpi.unit}"
                    logger.info(f"  With Unit: {formatted_value}")
            
            logger.info(f"  SUCCESS: {kpi.name} = {formatted_value}")
            
            return KPIResult(
                name=kpi.name,
                description=kpi.description,
                value=value,
                formatted_value=formatted_value,
                category=kpi.category,
                importance=kpi.importance,
                sql_query=kpi.sql_query
            )
            
        except Exception as e:
            logger.error(f"  ERROR computing KPI '{kpi.name}': {e}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return KPIResult(
                name=kpi.name,
                description=kpi.description,
                value=None,
                formatted_value="Error",
                category=kpi.category,
                importance=kpi.importance,
                sql_query=kpi.sql_query,
                error=str(e)
            )
    
    def _validate_query(self, query: str) -> bool:
        """Validate that a query is safe to execute."""
        if not query or not query.strip():
            return False
        
        query_upper = query.strip().upper()
        
        if not query_upper.startswith('SELECT'):
            return False
        
        dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
                     'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE']
        
        for keyword in dangerous:
            if f' {keyword} ' in f' {query_upper} ':
                return False
        
        return True
    
    def fix_query_with_llm(self, kpi: KPIDefinition, error: str, schema_name: str) -> Optional[str]:
        """Use LLM to fix a broken query."""
        if not self.client:
            return None
        
        logger.info(f"  Attempting to fix query with LLM...")
        logger.info(f"  Original Error: {error}")
        
        schema_text = self.db_manager.get_schema_for_llm(schema_name)
        
        prompt = f"""The following SQL query for KPI "{kpi.name}" failed with this error:

Error: {error}

Original Query:
{kpi.sql_query}

Database Schema:
{schema_text}

Please provide a corrected SQL query that:
1. Is a valid PostgreSQL SELECT statement
2. Returns a SINGLE scalar value (use aggregations)
3. Uses the correct table and column names from the schema
4. Handles NULL values with COALESCE
5. ROUNDS numeric results to 2 decimal places using ROUND(value::numeric, 2)

Return ONLY the corrected SQL query, nothing else."""

        logger.info(f"  LLM Fix Prompt:\n{prompt[:500]}...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Fix the query and return only the corrected SQL. Always ROUND numeric results to 2 decimal places."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            fixed_query = response.choices[0].message.content.strip()
            
            if fixed_query.startswith("```"):
                lines = fixed_query.split("\n")
                fixed_query = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            logger.info(f"  LLM Fixed Query: {fixed_query}")
            
            return fixed_query
            
        except Exception as e:
            logger.error(f"  Failed to fix query with LLM: {e}")
            return None
    
    def compute_kpis_with_retry(self, request: KPIComputeRequest, max_retries: int = 1) -> KPIComputeResponse:
        """Compute KPIs with automatic retry for failed queries."""
        logger.info("=" * 80)
        logger.info("DATA ENGINEER - Computing KPIs (with retry)")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Schema: {request.schema_name}")
        logger.info(f"Number of KPIs to compute: {len(request.kpis)}")
        logger.info(f"Max retries per KPI: {max_retries}")
        
        results = []
        success_count = 0
        error_count = 0
        
        for i, kpi in enumerate(request.kpis):
            logger.info(f"\n--- Computing KPI {i+1}/{len(request.kpis)}: {kpi.name} ---")
            result = self.compute_single_kpi(kpi, request.schema_name)
            
            if result.error and self.client and max_retries > 0:
                logger.info(f"  KPI failed, attempting LLM fix...")
                
                fixed_query = self.fix_query_with_llm(kpi, result.error, request.schema_name)
                
                if fixed_query:
                    fixed_kpi = KPIDefinition(
                        name=kpi.name,
                        description=kpi.description,
                        sql_query=fixed_query,
                        category=kpi.category,
                        importance=kpi.importance,
                        format_type=kpi.format_type,
                        unit=kpi.unit
                    )
                    
                    logger.info(f"  Retrying with fixed query...")
                    result = self.compute_single_kpi(fixed_kpi, request.schema_name)
                    
                    if not result.error:
                        logger.info(f"  SUCCESS: Query fix worked!")
                    else:
                        logger.error(f"  FAILED: Query fix did not resolve the issue")
            
            results.append(result)
            
            if result.error:
                error_count += 1
            else:
                success_count += 1
        
        logger.info(f"\n{'=' * 40}")
        logger.info(f"COMPUTATION SUMMARY")
        logger.info(f"{'=' * 40}")
        logger.info(f"Total KPIs: {len(request.kpis)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {error_count}")
        
        for r in results:
            status = "✓" if not r.error else "✗"
            logger.info(f"  {status} {r.name}: {r.formatted_value}")
        
        logger.info("=" * 80)
        
        return KPIComputeResponse(
            schema_name=request.schema_name,
            results=results,
            success_count=success_count,
            error_count=error_count
        )
    
    def close(self):
        """Clean up resources."""
        self.db_manager.close()
        self.tool_executor.close()
