"""Database manager for PostgreSQL operations."""

import re
import logging
from typing import Any, Optional
from datetime import datetime

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, RealDictCursor

from backend.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database operations for Excel data import."""
    
    # History table schema
    HISTORY_TABLE_SCHEMA = "public"
    HISTORY_TABLE_NAME = "_de_app_history"
    
    def __init__(self):
        """Initialize the database manager."""
        self.settings = get_settings()
        self._connection = None
    
    def _get_connection(self):
        """Get or create a database connection."""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                database=self.settings.postgres_db
            )
        return self._connection
    
    def close(self):
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    def rollback(self):
        """Rollback the current transaction."""
        if self._connection and not self._connection.closed:
            self._connection.rollback()
            logger.info("Transaction rolled back")
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Sanitize a name for use as schema/table/column name.
        
        - Convert to lowercase
        - Replace spaces and special chars with underscores
        - Remove leading/trailing underscores
        - Ensure it starts with a letter or underscore
        """
        # Remove file extension if present
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        
        # Convert to lowercase
        sanitized = name.lower()
        
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-z0-9_]', '_', sanitized)
        
        # Replace multiple underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        # If empty, use a default name
        if not sanitized:
            sanitized = '_unnamed'
        
        return sanitized
    
    def ensure_history_table(self):
        """Create the import history table if it doesn't exist."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.HISTORY_TABLE_SCHEMA}.{self.HISTORY_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(500) NOT NULL,
                    file_path VARCHAR(1000),
                    schema_name VARCHAR(255) NOT NULL,
                    table_name VARCHAR(255) NOT NULL,
                    rows_inserted INTEGER DEFAULT 0,
                    columns_standardized INTEGER DEFAULT 0,
                    column_mappings JSONB,
                    status VARCHAR(50) NOT NULL,
                    error_message TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def log_import(
        self,
        file_name: str,
        file_path: str,
        schema_name: str,
        table_name: str,
        rows_inserted: int,
        columns_standardized: int,
        column_mappings: Optional[dict],
        status: str,
        error_message: Optional[str] = None
    ):
        """Log an import operation to the history table."""
        self.ensure_history_table()
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.HISTORY_TABLE_SCHEMA}.{self.HISTORY_TABLE_NAME}
                (file_name, file_path, schema_name, table_name, rows_inserted, 
                 columns_standardized, column_mappings, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (file_name, file_path, schema_name, table_name, rows_inserted,
                 columns_standardized, psycopg2.extras.Json(column_mappings) if column_mappings else None,
                 status, error_message)
            )
            conn.commit()
    
    def get_import_history(self, limit: int = 100) -> list[dict]:
        """Get recent import history."""
        self.ensure_history_table()
        conn = self._get_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT * FROM {self.HISTORY_TABLE_SCHEMA}.{self.HISTORY_TABLE_NAME}
                ORDER BY processed_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            return [dict(row) for row in cur.fetchall()]
    
    def create_schema(self, schema_name: str) -> bool:
        """
        Create a schema if it doesn't exist.
        
        Returns True if schema was created, False if it already existed.
        """
        sanitized_name = self.sanitize_name(schema_name)
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            # Check if schema exists
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (sanitized_name,)
            )
            exists = cur.fetchone() is not None
            
            if not exists:
                cur.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(sanitized_name))
                )
                conn.commit()
                logger.info(f"Created schema: {sanitized_name}")
                return True
            
            logger.info(f"Schema already exists: {sanitized_name}")
            return False
    
    def get_table_columns(self, schema_name: str, table_name: str) -> list[dict[str, str]]:
        """Get column information for a table."""
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        conn = self._get_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (sanitized_schema, sanitized_table)
            )
            return [dict(row) for row in cur.fetchall()]
    
    def table_exists(self, schema_name: str, table_name: str) -> bool:
        """Check if a table exists."""
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (sanitized_schema, sanitized_table)
            )
            return cur.fetchone() is not None
    
    def create_table(
        self,
        schema_name: str,
        table_name: str,
        columns: dict[str, str]
    ) -> bool:
        """
        Create a table with the specified columns.
        
        Args:
            schema_name: Name of the schema
            table_name: Name of the table
            columns: Dictionary mapping column names to PostgreSQL types
            
        Returns:
            True if table was created, False if it already existed
        """
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        
        if self.table_exists(schema_name, table_name):
            logger.info(f"Table already exists: {sanitized_schema}.{sanitized_table}")
            return False
        
        conn = self._get_connection()
        
        # Build column definitions
        col_defs = []
        for col_name, col_type in columns.items():
            sanitized_col = self.sanitize_name(col_name)
            col_defs.append(
                sql.SQL("{} {}").format(
                    sql.Identifier(sanitized_col),
                    sql.SQL(col_type)
                )
            )
        
        # Add auto-increment ID column
        col_defs.insert(0, sql.SQL("_de_row_id SERIAL PRIMARY KEY"))
        
        # Add import timestamp column
        col_defs.append(sql.SQL("_de_imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
        
        with conn.cursor() as cur:
            create_query = sql.SQL("CREATE TABLE {}.{} ({})").format(
                sql.Identifier(sanitized_schema),
                sql.Identifier(sanitized_table),
                sql.SQL(", ").join(col_defs)
            )
            cur.execute(create_query)
            conn.commit()
        
        logger.info(f"Created table: {sanitized_schema}.{sanitized_table}")
        return True
    
    def add_columns(
        self,
        schema_name: str,
        table_name: str,
        new_columns: dict[str, str]
    ):
        """Add new columns to an existing table."""
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        conn = self._get_connection()
        
        existing_cols = {col['column_name'] for col in self.get_table_columns(schema_name, table_name)}
        
        with conn.cursor() as cur:
            for col_name, col_type in new_columns.items():
                sanitized_col = self.sanitize_name(col_name)
                if sanitized_col not in existing_cols:
                    alter_query = sql.SQL("ALTER TABLE {}.{} ADD COLUMN {} {}").format(
                        sql.Identifier(sanitized_schema),
                        sql.Identifier(sanitized_table),
                        sql.Identifier(sanitized_col),
                        sql.SQL(col_type)
                    )
                    cur.execute(alter_query)
                    logger.info(f"Added column {sanitized_col} to {sanitized_schema}.{sanitized_table}")
            
            conn.commit()
    
    def insert_data(
        self,
        schema_name: str,
        table_name: str,
        columns: list[str],
        data: list[list[Any]]
    ) -> int:
        """
        Insert data into a table.
        
        Args:
            schema_name: Name of the schema
            table_name: Name of the table
            columns: List of column names
            data: List of rows (each row is a list of values)
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        sanitized_cols = [self.sanitize_name(col) for col in columns]
        
        conn = self._get_connection()
        
        # Build the INSERT query
        col_identifiers = sql.SQL(", ").join([sql.Identifier(col) for col in sanitized_cols])
        
        insert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
            sql.Identifier(sanitized_schema),
            sql.Identifier(sanitized_table),
            col_identifiers
        )
        
        with conn.cursor() as cur:
            # Convert data to tuples
            tuple_data = [tuple(row) for row in data]
            execute_values(cur, insert_query.as_string(conn), tuple_data)
            rows_inserted = cur.rowcount
            conn.commit()
        
        logger.info(f"Inserted {rows_inserted} rows into {sanitized_schema}.{sanitized_table}")
        return rows_inserted
    
    def get_all_schemas(self) -> list[str]:
        """Get all user-created schemas (excluding system schemas)."""
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT schema_name 
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                AND schema_name NOT LIKE 'pg_%'
                ORDER BY schema_name
            """)
            return [row[0] for row in cur.fetchall()]
    
    def get_schema_tables(self, schema_name: str) -> list[str]:
        """Get all tables in a schema."""
        sanitized_schema = self.sanitize_name(schema_name)
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name 
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
                """,
                (sanitized_schema,)
            )
            return [row[0] for row in cur.fetchall()]
    
    def get_table_row_count(self, schema_name: str, table_name: str) -> int:
        """Get the row count for a table."""
        sanitized_schema = self.sanitize_name(schema_name)
        sanitized_table = self.sanitize_name(table_name)
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(sanitized_schema),
                    sql.Identifier(sanitized_table)
                )
            )
            return cur.fetchone()[0]
    
    def get_all_columns_in_schema(self, schema_name: str) -> dict[str, list[str]]:
        """
        Get all columns for all tables in a schema.
        
        Returns a dictionary mapping table names to lists of column names.
        """
        sanitized_schema = self.sanitize_name(schema_name)
        conn = self._get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                AND column_name NOT LIKE '_de_%'
                ORDER BY table_name, ordinal_position
                """,
                (sanitized_schema,)
            )
            
            result = {}
            for table_name, column_name in cur.fetchall():
                if table_name not in result:
                    result[table_name] = []
                result[table_name].append(column_name)
            
            return result
    
    def infer_postgres_type(self, values: list[Any]) -> str:
        """
        Infer the PostgreSQL type from a list of values.
        
        Args:
            values: Sample values from a column
            
        Returns:
            PostgreSQL type string
        """
        # Filter out None values
        non_null_values = [v for v in values if v is not None and str(v).strip() != '']
        
        if not non_null_values:
            return "TEXT"
        
        # Check for boolean
        bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f'}
        if all(str(v).lower() in bool_values for v in non_null_values):
            return "BOOLEAN"
        
        # Check for integer
        try:
            for v in non_null_values:
                if isinstance(v, bool):
                    raise ValueError()
                int_val = int(float(str(v)))
                if float(str(v)) != int_val:
                    raise ValueError()
            return "BIGINT"
        except (ValueError, TypeError):
            pass
        
        # Check for float
        try:
            for v in non_null_values:
                float(str(v))
            return "DOUBLE PRECISION"
        except (ValueError, TypeError):
            pass
        
        # Check for date/datetime
        from datetime import datetime as dt
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                for v in non_null_values:
                    if isinstance(v, (dt,)):
                        continue
                    dt.strptime(str(v), fmt)
                return "TIMESTAMP" if ' ' in date_formats[0] else "DATE"
            except (ValueError, TypeError):
                continue
        
        # Check for datetime objects directly
        if all(isinstance(v, dt) for v in non_null_values):
            return "TIMESTAMP"
        
        # Default to TEXT for safety - VARCHAR can cause issues with variable length data
        # TEXT has no length limit and performs similarly in PostgreSQL
        return "TEXT"

