"""Pydantic models for KPI definitions and results."""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class KPICategory(str, Enum):
    """Categories for KPIs."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    PRODUCT = "product"
    PERFORMANCE = "performance"
    GROWTH = "growth"
    QUALITY = "quality"
    OTHER = "other"


class KPIImportance(str, Enum):
    """Importance levels for KPIs."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class KPIDefinition(BaseModel):
    """Definition of a KPI suggested by the analyst agent."""
    name: str = Field(..., description="Human-readable name of the KPI")
    description: str = Field(..., description="What this KPI measures and why it matters")
    sql_query: str = Field(..., description="SQL query to compute this KPI")
    category: KPICategory = Field(default=KPICategory.OTHER, description="Category of the KPI")
    importance: KPIImportance = Field(default=KPIImportance.MEDIUM, description="Importance level")
    format_type: str = Field(default="auto", description="How to format the result (auto, currency, percentage, number)")
    unit: Optional[str] = Field(default=None, description="Unit of measurement (e.g., $, %, users)")


class KPIResult(BaseModel):
    """Result of computing a KPI."""
    name: str
    description: str
    value: Any
    formatted_value: str
    category: KPICategory
    importance: KPIImportance
    sql_query: str
    computed_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


class KPIAnalysisRequest(BaseModel):
    """Request to analyze a schema and suggest KPIs."""
    schema_name: str = Field(..., description="Name of the database schema to analyze")
    company_name: str = Field(..., description="Name of the company")
    company_description: Optional[str] = Field(default=None, description="Description of what the company does")
    industry: Optional[str] = Field(default=None, description="Industry/vertical of the company")
    max_kpis: int = Field(default=10, ge=1, le=20, description="Maximum number of KPIs to suggest")


class KPIAnalysisResponse(BaseModel):
    """Response containing suggested KPIs."""
    schema_name: str
    company_name: str
    kpis: list[KPIDefinition]
    analysis_summary: str = Field(..., description="Summary of the data analysis")
    generated_at: datetime = Field(default_factory=datetime.now)


class KPIComputeRequest(BaseModel):
    """Request to compute specific KPIs."""
    schema_name: str
    kpis: list[KPIDefinition]


class KPIComputeResponse(BaseModel):
    """Response containing computed KPI values."""
    schema_name: str
    results: list[KPIResult]
    computed_at: datetime = Field(default_factory=datetime.now)
    success_count: int
    error_count: int


class KPIDashboard(BaseModel):
    """Complete KPI dashboard for a company."""
    schema_name: str
    company_name: str
    company_description: Optional[str] = None
    industry: Optional[str] = None
    kpis: list[KPIResult]
    generated_at: datetime
    last_computed_at: datetime


class CompanyContext(BaseModel):
    """Context about a company for KPI analysis."""
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to a prompt-friendly string."""
        parts = [f"Company: {self.name}"]
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)

