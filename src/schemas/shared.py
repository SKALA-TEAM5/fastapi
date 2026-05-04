from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class AuditResult(BaseModel):
    is_compliant: bool = Field(description="산안비 법령 기준에 적합하면 True")
    confidence: float = Field(description="판정 확신도 (0.0~1.0)")
    reasoning: str = Field(description="판정 근거 (법령 조항 인용 포함)")
    referenced_laws: List[str] = Field(description="참조한 법령 조항 목록")
    evidence_snippets: List[str] = Field(default=[])
    needs_human_review: bool = Field(default=False)
    top_source: str = Field(default="")


class AgenticRAGState(TypedDict):
    question: str
    documents: List[Document]
    judgment: Optional[dict]
    retry_count: int


class AuditPipelineState(BaseModel):
    question: str
    context: str
    evidence_snippets: List[str] = []
    evidence_laws: List[str] = []
    result: Optional[AuditResult] = None
