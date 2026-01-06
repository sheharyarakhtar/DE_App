"""KPI Analyst Agent - Suggests relevant KPIs based on schema and company context."""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from openai import OpenAI

from backend.config import get_settings
from backend.services.db_manager import DatabaseManager
from backend.services.agent_tools import AgentToolExecutor, SCHEMA_TOOLS
from backend.models.kpi_schemas import (
    KPIDefinition,
    KPIAnalysisRequest,
    KPIAnalysisResponse,
    KPICategory,
    KPIImportance,
    CompanyContext
)

# Set up file logging for KPI operations
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "kpi_analysis.log")

kpi_logger = logging.getLogger("kpi_analyst")
kpi_logger.setLevel(logging.DEBUG)

# File handler - logs to file (will be recreated on reset)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
kpi_logger.addHandler(file_handler)

# Also keep console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
kpi_logger.addHandler(console_handler)

logger = kpi_logger


def reset_kpi_logs():
    """Clear all KPI log files for a fresh dashboard generation."""
    # Clear kpi_analysis.log
    analysis_log = os.path.join(LOG_DIR, "kpi_analysis.log")
    if os.path.exists(analysis_log):
        open(analysis_log, 'w').close()
    
    # Clear kpi_computation.log
    computation_log = os.path.join(LOG_DIR, "kpi_computation.log")
    if os.path.exists(computation_log):
        open(computation_log, 'w').close()
    
    logger.info("=" * 80)
    logger.info("KPI LOGS RESET - Starting fresh dashboard generation")
    logger.info("=" * 80)


def get_analyst_system_prompt(company_name: str, industry: str, description: str) -> str:
    """Generate a context-aware system prompt based on company info."""
    
    industry_context = ""
    if industry:
        industry_kpis = {
            "ecommerce": "conversion rates, average order value, cart abandonment rate, customer lifetime value, return rate",
            "saas": "monthly recurring revenue (MRR), churn rate, customer acquisition cost (CAC), lifetime value (LTV), active users",
            "finance": "transaction volume, default rates, portfolio performance, risk metrics, compliance rates",
            "healthcare": "patient outcomes, readmission rates, treatment effectiveness, wait times, resource utilization",
            "manufacturing": "production efficiency, defect rates, inventory turnover, equipment uptime, cycle time",
            "services": "utilization rates, project profitability, client satisfaction, billable hours, resource allocation",
            "education": "completion rates, student performance, engagement metrics, retention rates",
            "other": "operational efficiency, quality metrics, performance indicators"
        }
        industry_kpis_examples = industry_kpis.get(industry, industry_kpis["other"])
        industry_context = f"""
## Industry Context: {industry.upper()}
For {industry} businesses, typical important KPIs include: {industry_kpis_examples}.
Consider these types of metrics when analyzing the data, but adapt based on what's actually available."""

    company_context = ""
    if description:
        company_context = f"""
## Company Context
{description}

Use this context to understand what metrics would be most valuable. Focus on KPIs that directly relate to the company's core business activities described above."""

    return f"""You are a Senior Data Analytics Expert specializing in business intelligence and KPI development for {company_name}.

Your role is to analyze database schemas and suggest HIGHLY RELEVANT, ACTIONABLE KPIs that will provide real business value.
{industry_context}
{company_context}

## Your Analysis Process:
1. First, use get_schema_summary to understand all available tables and columns
2. Use get_sample_data on the MOST IMPORTANT tables to understand actual data values and patterns
3. Use get_column_statistics on numeric columns that could be meaningful KPIs
4. Based on your analysis AND the company context, suggest 5-10 KPIs that would genuinely help this business

## Critical KPI Guidelines:
- KPIs MUST be directly relevant to the company's business (use the description provided)
- KPIs should answer real business questions like "How are we performing?" "Where are issues?"
- Each KPI should be ACTIONABLE - it should inform a decision
- Avoid generic counts unless they're meaningful (e.g., "Total Tasks Reviewed" is better than just "Row Count")
- For percentage/rate KPIs, calculate them properly (e.g., problematic_count / total_count * 100)
- Name KPIs clearly - a business user should understand what it measures

## SQL Query Requirements - CRITICAL:
- Use fully qualified table names: schema_name.table_name
- **EVERY query MUST return EXACTLY ONE ROW with ONE VALUE** - this is non-negotiable
- Use aggregations: COUNT, SUM, AVG, MAX, MIN - these return single values
- **NEVER use GROUP BY** - it returns multiple rows which breaks our dashboard
- **NEVER return multiple columns** - only one value per KPI
- Handle NULL values with COALESCE
- For percentages, multiply by 100 and round appropriately
- For averages and rates, ROUND to 2 decimal places

WRONG (returns multiple rows):
  SELECT trainer_name, COUNT(*) FROM table GROUP BY trainer_name
  
CORRECT (returns single value):
  SELECT COUNT(DISTINCT trainer_name) FROM table  -- counts unique trainers
  SELECT ROUND(AVG(score), 2) FROM table  -- average across all

## Output Format:
Provide your response as a JSON object:
{{
    "analysis_summary": "What data is available and what business insights it can provide",
    "kpis": [
        {{
            "name": "Clear Business-Focused Name",
            "description": "What this measures and WHY it matters for this specific business",
            "sql_query": "SELECT ROUND(AVG(column), 2) FROM schema.table",
            "category": "financial|operational|customer|product|performance|growth|quality|other",
            "importance": "high|medium|low",
            "format_type": "auto|currency|percentage|number",
            "unit": "tasks|reviews|%|$|etc or null"
        }}
    ]
}}

Remember: Quality over quantity. 5 excellent, relevant KPIs are better than 10 generic ones."""


class KPIAnalyst:
    """Agent that analyzes schemas and suggests relevant KPIs."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.tool_executor = AgentToolExecutor()
        
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. KPI Analyst will not work.")
    
    def analyze(self, request: KPIAnalysisRequest) -> KPIAnalysisResponse:
        """
        Analyze a schema and suggest relevant KPIs.
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        logger.info("=" * 80)
        logger.info("KPI ANALYST - Starting Analysis")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Schema: {request.schema_name}")
        logger.info(f"Company: {request.company_name}")
        logger.info(f"Industry: {request.industry}")
        logger.info(f"Description: {request.company_description}")
        logger.info(f"Max KPIs: {request.max_kpis}")
        
        # Build context-aware system prompt
        system_prompt = get_analyst_system_prompt(
            request.company_name,
            request.industry or "other",
            request.company_description or ""
        )
        
        logger.info("-" * 40)
        logger.info("SYSTEM PROMPT:")
        logger.info("-" * 40)
        logger.info(system_prompt)
        
        # Get schema info for context
        schema_text = self.db_manager.get_schema_for_llm(request.schema_name)
        
        logger.info("-" * 40)
        logger.info("SCHEMA INFO:")
        logger.info("-" * 40)
        logger.info(schema_text)
        
        # Build focused user prompt
        user_prompt = f"""Analyze the database for {request.company_name} and suggest {request.max_kpis} highly relevant KPIs.

Company: {request.company_name}
Industry: {request.industry or 'Not specified'}
Description: {request.company_description or 'Not provided'}

Available Database Schema:
{schema_text}

Instructions:
1. Use the tools to explore the data (get_schema_summary, get_sample_data, get_column_statistics)
2. Focus on metrics that would matter for THIS specific business based on the description
3. Ensure SQL queries return properly rounded single values
4. Provide your final KPI suggestions in the required JSON format"""

        logger.info("-" * 40)
        logger.info("USER PROMPT:")
        logger.info("-" * 40)
        logger.info(user_prompt)

        # Run the agent loop
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n--- Agent Iteration {iteration}/{max_iterations} ---")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=SCHEMA_TOOLS,
                tool_choice="auto",
                temperature=0.5  # Lower temperature for more focused responses
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            
            logger.info(f"LLM Response - Finish Reason: {response.choices[0].finish_reason}")
            logger.info(f"LLM Response - Has Tool Calls: {bool(assistant_message.tool_calls)}")
            
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"\n  TOOL CALL: {tool_name}")
                    logger.info(f"  ARGUMENTS: {json.dumps(tool_args, indent=2)}")
                    
                    result = self.tool_executor.execute_tool(tool_name, tool_args)
                    
                    result_preview = result[:2000] + "..." if len(result) > 2000 else result
                    logger.info(f"  TOOL RESULT:\n{result_preview}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                logger.info("\n--- Agent Complete - Final Response ---")
                logger.info(f"FINAL LLM RESPONSE:\n{assistant_message.content}")
                break
        
        result = self._parse_response(
            assistant_message.content,
            request.schema_name,
            request.company_name
        )
        
        logger.info("-" * 40)
        logger.info(f"PARSED RESULT: {len(result.kpis)} KPIs extracted")
        logger.info(f"Analysis Summary: {result.analysis_summary}")
        for i, kpi in enumerate(result.kpis):
            logger.info(f"  KPI {i+1}: {kpi.name}")
            logger.info(f"    Category: {kpi.category}")
            logger.info(f"    SQL: {kpi.sql_query}")
        logger.info("=" * 80)
        
        return result
    
    def _parse_response(
        self, 
        content: str, 
        schema_name: str, 
        company_name: str
    ) -> KPIAnalysisResponse:
        """Parse the agent's response into a structured format."""
        logger.info("Parsing LLM response...")
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                logger.info(f"Extracted JSON (length: {len(json_str)})")
                data = json.loads(json_str)
            else:
                logger.info("No JSON block found, trying to parse entire content")
                data = json.loads(content)
            
            kpis = []
            for kpi_data in data.get("kpis", []):
                try:
                    kpi = KPIDefinition(
                        name=kpi_data.get("name", "Unnamed KPI"),
                        description=kpi_data.get("description", ""),
                        sql_query=kpi_data.get("sql_query", ""),
                        category=KPICategory(kpi_data.get("category", "other")),
                        importance=KPIImportance(kpi_data.get("importance", "medium")),
                        format_type=kpi_data.get("format_type", "auto"),
                        unit=kpi_data.get("unit")
                    )
                    kpis.append(kpi)
                    logger.info(f"  Parsed KPI: {kpi.name}")
                except Exception as e:
                    logger.warning(f"Failed to parse KPI: {e} - Data: {kpi_data}")
                    continue
            
            return KPIAnalysisResponse(
                schema_name=schema_name,
                company_name=company_name,
                kpis=kpis,
                analysis_summary=data.get("analysis_summary", "Analysis complete.")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent response as JSON: {e}")
            logger.error(f"Raw content: {content}")
            return KPIAnalysisResponse(
                schema_name=schema_name,
                company_name=company_name,
                kpis=[],
                analysis_summary=f"Failed to parse KPI suggestions. Raw response: {content[:500]}"
            )
    
    def suggest_basic_kpis(self, schema_name: str) -> list[KPIDefinition]:
        """
        Suggest basic KPIs without using LLM (fallback).
        """
        logger.info("=" * 80)
        logger.info("KPI ANALYST - Generating Basic KPIs (No LLM)")
        logger.info("=" * 80)
        logger.info(f"Schema: {schema_name}")
        
        summary = self.db_manager.get_schema_summary(schema_name)
        kpis = []
        
        logger.info(f"Found {len(summary['tables'])} tables")
        
        total_tables = len(summary["tables"])
        if total_tables > 0:
            total_rows = sum(t["row_count"] for t in summary["tables"])
            kpi = KPIDefinition(
                name="Total Records",
                description="Total number of records across all tables in the database",
                sql_query=f"SELECT {total_rows} as total_records",
                category=KPICategory.OPERATIONAL,
                importance=KPIImportance.MEDIUM,
                format_type="number"
            )
            kpis.append(kpi)
            logger.info(f"  Added KPI: {kpi.name}")
        
        for table in summary["tables"][:5]:
            table_name = table["table_name"]
            kpi = KPIDefinition(
                name=f"{table_name.replace('_', ' ').title()} Count",
                description=f"Total number of records in {table_name}",
                sql_query=f"SELECT COUNT(*) FROM {schema_name}.{table_name}",
                category=KPICategory.OPERATIONAL,
                importance=KPIImportance.LOW,
                format_type="number"
            )
            kpis.append(kpi)
            logger.info(f"  Added KPI: {kpi.name}")
        
        for table in summary["tables"]:
            for col in table["columns"]:
                if col["data_type"] in ("integer", "bigint", "numeric", "double precision", "real"):
                    col_name = col["column_name"]
                    table_name = table["table_name"]
                    
                    if 'id' in col_name.lower():
                        continue
                    
                    # Use ROUND for numeric aggregations
                    kpi = KPIDefinition(
                        name=f"Total {col_name.replace('_', ' ').title()}",
                        description=f"Sum of {col_name} in {table_name}",
                        sql_query=f"SELECT ROUND(COALESCE(SUM({col_name}), 0)::numeric, 2) FROM {schema_name}.{table_name}",
                        category=KPICategory.FINANCIAL,
                        importance=KPIImportance.MEDIUM,
                        format_type="number"
                    )
                    kpis.append(kpi)
                    logger.info(f"  Added KPI: {kpi.name}")
                    
                    kpi = KPIDefinition(
                        name=f"Average {col_name.replace('_', ' ').title()}",
                        description=f"Average of {col_name} in {table_name}",
                        sql_query=f"SELECT ROUND(COALESCE(AVG({col_name}), 0)::numeric, 2) FROM {schema_name}.{table_name}",
                        category=KPICategory.PERFORMANCE,
                        importance=KPIImportance.MEDIUM,
                        format_type="number"
                    )
                    kpis.append(kpi)
                    logger.info(f"  Added KPI: {kpi.name}")
                    
                    if len(kpis) >= 10:
                        break
            
            if len(kpis) >= 10:
                break
        
        logger.info(f"Generated {len(kpis)} basic KPIs")
        logger.info("=" * 80)
        
        return kpis[:10]
    
    def close(self):
        """Clean up resources."""
        self.db_manager.close()
        self.tool_executor.close()
