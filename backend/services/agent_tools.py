"""Shared tool definitions for KPI agents using OpenAI function calling."""

import json
import logging
from typing import Any, Callable

from backend.services.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# Tool definitions for OpenAI function calling
SCHEMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema_summary",
            "description": "Get a comprehensive summary of a database schema including all tables, their columns with data types, and row counts. Use this to understand what data is available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "The name of the database schema to analyze"
                    }
                },
                "required": ["schema_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_data",
            "description": "Get sample rows from a specific table to understand the actual data values and patterns. Useful for understanding what kind of data each column contains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "The name of the database schema"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to sample"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of sample rows to return (default 5, max 20)",
                        "default": 5
                    }
                },
                "required": ["schema_name", "table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_statistics",
            "description": "Get statistics for a specific column including min, max, average (for numeric columns), distinct count, null count, and top values. Use this to understand data distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "The name of the database schema"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table"
                    },
                    "column_name": {
                        "type": "string",
                        "description": "The name of the column to analyze"
                    }
                },
                "required": ["schema_name", "table_name", "column_name"]
            }
        }
    }
]

QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute a read-only SQL SELECT query against the database. Only SELECT statements are allowed. Use this to compute KPI values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute"
                    },
                    "schema_name": {
                        "type": "string",
                        "description": "The schema to set as search path (optional)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Combined tools for agents that need both
ALL_TOOLS = SCHEMA_TOOLS + QUERY_TOOLS


class AgentToolExecutor:
    """Executes tool calls from OpenAI agents."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
        # Map tool names to methods
        self.tool_handlers: dict[str, Callable] = {
            "get_schema_summary": self._handle_get_schema_summary,
            "get_sample_data": self._handle_get_sample_data,
            "get_column_statistics": self._handle_get_column_statistics,
            "execute_sql_query": self._handle_execute_sql_query,
        }
    
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool and return the result as a JSON string.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            JSON string with the result or error
        """
        handler = self.tool_handlers.get(tool_name)
        
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        try:
            result = handler(arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return json.dumps({"error": str(e)})
    
    def _handle_get_schema_summary(self, args: dict) -> dict:
        """Handle get_schema_summary tool call."""
        schema_name = args.get("schema_name")
        if not schema_name:
            return {"error": "schema_name is required"}
        
        return self.db_manager.get_schema_summary(schema_name)
    
    def _handle_get_sample_data(self, args: dict) -> dict:
        """Handle get_sample_data tool call."""
        schema_name = args.get("schema_name")
        table_name = args.get("table_name")
        limit = min(args.get("limit", 5), 20)  # Cap at 20
        
        if not schema_name or not table_name:
            return {"error": "schema_name and table_name are required"}
        
        rows = self.db_manager.get_sample_data(schema_name, table_name, limit)
        return {"table": table_name, "sample_rows": rows, "count": len(rows)}
    
    def _handle_get_column_statistics(self, args: dict) -> dict:
        """Handle get_column_statistics tool call."""
        schema_name = args.get("schema_name")
        table_name = args.get("table_name")
        column_name = args.get("column_name")
        
        if not all([schema_name, table_name, column_name]):
            return {"error": "schema_name, table_name, and column_name are required"}
        
        return self.db_manager.get_column_statistics(schema_name, table_name, column_name)
    
    def _handle_execute_sql_query(self, args: dict) -> dict:
        """Handle execute_sql_query tool call."""
        query = args.get("query")
        schema_name = args.get("schema_name")
        
        if not query:
            return {"error": "query is required"}
        
        return self.db_manager.execute_read_query(query, schema_name)
    
    def close(self):
        """Close database connection."""
        self.db_manager.close()


def format_tool_result(result: Any, format_type: str = "auto") -> str:
    """
    Format a tool result for display.
    
    Args:
        result: The result to format
        format_type: How to format (auto, currency, percentage, number, text)
        
    Returns:
        Formatted string
    """
    if result is None:
        return "N/A"
    
    if format_type == "auto":
        if isinstance(result, bool):
            return "Yes" if result else "No"
        elif isinstance(result, float):
            if abs(result) >= 1000000:
                return f"{result/1000000:.2f}M"
            elif abs(result) >= 1000:
                return f"{result/1000:.2f}K"
            elif abs(result) < 1:
                return f"{result:.4f}"
            else:
                return f"{result:.2f}"
        elif isinstance(result, int):
            if abs(result) >= 1000000:
                return f"{result/1000000:.2f}M"
            elif abs(result) >= 1000:
                return f"{result/1000:.2f}K"
            else:
                return str(result)
        else:
            return str(result)
    
    elif format_type == "currency":
        if isinstance(result, (int, float)):
            if abs(result) >= 1000000:
                return f"${result/1000000:.2f}M"
            elif abs(result) >= 1000:
                return f"${result/1000:.2f}K"
            else:
                return f"${result:.2f}"
        return str(result)
    
    elif format_type == "percentage":
        if isinstance(result, (int, float)):
            return f"{result:.1f}%"
        return str(result)
    
    elif format_type == "number":
        if isinstance(result, (int, float)):
            return f"{result:,.0f}"
        return str(result)
    
    else:
        return str(result)

