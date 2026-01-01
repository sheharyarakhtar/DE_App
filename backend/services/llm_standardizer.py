"""LLM-powered column standardization service using OpenAI."""

import json
import logging
import hashlib
from typing import Optional

from openai import OpenAI

from backend.config import get_settings
from backend.services.db_manager import DatabaseManager
from backend.models.schemas import ColumnMapping, StandardizationResult

logger = logging.getLogger(__name__)


class LLMStandardizer:
    """
    Uses OpenAI to standardize column names for consistency across imports.
    
    This service analyzes incoming Excel column names and compares them against
    existing database columns to suggest standardized names.
    """
    
    # Cache for column mappings to reduce API calls
    _mapping_cache: dict[str, dict[str, str]] = {}
    
    def __init__(self):
        """Initialize the LLM standardizer."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        
        # Initialize OpenAI client if API key is available
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. Column standardization will be disabled.")
    
    def _get_cache_key(
        self,
        schema_name: str,
        table_name: str,
        columns: list[str]
    ) -> str:
        """Generate a cache key for a set of columns."""
        key_data = f"{schema_name}:{table_name}:{','.join(sorted(columns))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_existing_columns(self, schema_name: str) -> dict[str, list[str]]:
        """Get all existing columns in a schema."""
        try:
            return self.db_manager.get_all_columns_in_schema(schema_name)
        except Exception as e:
            logger.error(f"Error getting existing columns: {e}")
            return {}
    
    def standardize_columns(
        self,
        schema_name: str,
        table_name: str,
        incoming_columns: list[str],
        use_cache: bool = True
    ) -> StandardizationResult:
        """
        Standardize column names using LLM.
        
        Args:
            schema_name: Target schema name
            table_name: Target table name
            incoming_columns: List of column names from the Excel file
            use_cache: Whether to use cached mappings
            
        Returns:
            StandardizationResult with column mappings
        """
        # Check cache first
        cache_key = self._get_cache_key(schema_name, table_name, incoming_columns)
        if use_cache and cache_key in self._mapping_cache:
            logger.info(f"Using cached column mappings for {schema_name}.{table_name}")
            cached_mappings = self._mapping_cache[cache_key]
            return StandardizationResult(
                mappings=[
                    ColumnMapping(
                        original_name=orig,
                        standardized_name=std,
                        confidence=1.0,
                        reason="cached"
                    )
                    for orig, std in cached_mappings.items()
                ],
                schema_name=schema_name,
                table_name=table_name
            )
        
        # If no OpenAI client, return identity mapping with basic sanitization
        if not self.client:
            return self._basic_standardization(schema_name, table_name, incoming_columns)
        
        # Get existing columns in the schema
        existing_columns = self._get_existing_columns(schema_name)
        
        # Build the prompt
        prompt = self._build_prompt(
            schema_name,
            table_name,
            incoming_columns,
            existing_columns
        )
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = self._parse_response(
                response.choices[0].message.content,
                schema_name,
                table_name,
                incoming_columns
            )
            
            # Cache the result
            if use_cache:
                self._mapping_cache[cache_key] = {
                    m.original_name: m.standardized_name
                    for m in result.mappings
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fall back to basic standardization
            return self._basic_standardization(schema_name, table_name, incoming_columns)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for column standardization."""
        return """You are a database column name standardization expert. Your job is to:

1. Standardize column names to be consistent, clean, and database-friendly
2. Match incoming columns to existing columns when they represent the same data
3. Follow these naming conventions:
   - Use snake_case (lowercase with underscores)
   - Remove special characters
   - Use common abbreviations consistently (id, num, qty, amt, etc.)
   - Expand unclear abbreviations when helpful

Common standardizations:
- ID, Id, id_no, ID_No, identification → id
- Customer Name, customer_name, cust_name, CustomerName → customer_name
- Date, DATE, date_time, DateTime → date or created_at/updated_at as appropriate
- Amount, AMT, Amt, amount_value → amount
- Quantity, QTY, Qty, qty_ordered → quantity
- Email, EMAIL, email_address, e_mail → email
- Phone, PHONE, phone_number, phone_no → phone
- Address, ADDRESS, addr, address_line → address

Always respond with valid JSON in this format:
{
    "mappings": [
        {
            "original": "Original Column Name",
            "standardized": "standardized_column_name",
            "confidence": 0.95,
            "reason": "Brief explanation"
        }
    ]
}"""
    
    def _build_prompt(
        self,
        schema_name: str,
        table_name: str,
        incoming_columns: list[str],
        existing_columns: dict[str, list[str]]
    ) -> str:
        """Build the prompt for column standardization."""
        prompt_parts = [
            f"Schema: {schema_name}",
            f"Table: {table_name}",
            f"\nIncoming columns from Excel file:",
            json.dumps(incoming_columns, indent=2)
        ]
        
        if existing_columns:
            prompt_parts.append("\nExisting columns in the database schema:")
            for tbl, cols in existing_columns.items():
                prompt_parts.append(f"\n  Table '{tbl}': {', '.join(cols)}")
        else:
            prompt_parts.append("\nNo existing tables in this schema yet.")
        
        prompt_parts.append(
            "\n\nPlease standardize the incoming column names. "
            "If any incoming columns match existing columns semantically, "
            "use the existing column name for consistency."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_response(
        self,
        response_content: str,
        schema_name: str,
        table_name: str,
        incoming_columns: list[str]
    ) -> StandardizationResult:
        """Parse the OpenAI response into a StandardizationResult."""
        try:
            data = json.loads(response_content)
            mappings = []
            
            # Create a set of processed columns
            processed = set()
            
            for mapping in data.get("mappings", []):
                original = mapping.get("original", "")
                standardized = mapping.get("standardized", "")
                confidence = mapping.get("confidence", 1.0)
                reason = mapping.get("reason", "")
                
                if original and standardized:
                    mappings.append(ColumnMapping(
                        original_name=original,
                        standardized_name=DatabaseManager.sanitize_name(standardized),
                        confidence=confidence,
                        reason=reason
                    ))
                    processed.add(original)
            
            # Add any missing columns with basic sanitization
            for col in incoming_columns:
                if col not in processed:
                    mappings.append(ColumnMapping(
                        original_name=col,
                        standardized_name=DatabaseManager.sanitize_name(col),
                        confidence=1.0,
                        reason="not in LLM response, using basic sanitization"
                    ))
            
            return StandardizationResult(
                mappings=mappings,
                schema_name=schema_name,
                table_name=table_name
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            return self._basic_standardization(schema_name, table_name, incoming_columns)
    
    def _basic_standardization(
        self,
        schema_name: str,
        table_name: str,
        incoming_columns: list[str]
    ) -> StandardizationResult:
        """
        Perform basic column standardization without LLM.
        
        Uses simple rules and the sanitize_name function.
        """
        mappings = []
        
        for col in incoming_columns:
            standardized = DatabaseManager.sanitize_name(col)
            
            # Apply some common transformations
            standardized = self._apply_common_transformations(standardized)
            
            mappings.append(ColumnMapping(
                original_name=col,
                standardized_name=standardized,
                confidence=0.8,
                reason="basic sanitization (LLM not available)"
            ))
        
        return StandardizationResult(
            mappings=mappings,
            schema_name=schema_name,
            table_name=table_name
        )
    
    def _apply_common_transformations(self, name: str) -> str:
        """Apply common column name transformations."""
        # Common abbreviation expansions/standardizations
        transformations = {
            'id_no': 'id',
            'id_number': 'id',
            'identification': 'id',
            'cust_name': 'customer_name',
            'cust_id': 'customer_id',
            'prod_name': 'product_name',
            'prod_id': 'product_id',
            'qty': 'quantity',
            'amt': 'amount',
            'num': 'number',
            'desc': 'description',
            'addr': 'address',
            'tel': 'phone',
            'telephone': 'phone',
            'phone_no': 'phone',
            'phone_number': 'phone',
            'email_address': 'email',
            'e_mail': 'email',
            'dt': 'date',
            'date_time': 'datetime',
        }
        
        return transformations.get(name, name)
    
    def get_mapping_dict(self, result: StandardizationResult) -> dict[str, str]:
        """Convert StandardizationResult to a simple mapping dictionary."""
        return {m.original_name: m.standardized_name for m in result.mappings}
    
    def clear_cache(self):
        """Clear the mapping cache."""
        self._mapping_cache.clear()
        logger.info("Column mapping cache cleared")
    
    def close(self):
        """Clean up resources."""
        self.db_manager.close()

