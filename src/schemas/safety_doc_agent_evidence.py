from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvidenceRequirementItemContext:
    """AI가 필수 증빙을 판단할 때 쓰는 항목 문맥 뷰 응답."""

    project_id: int
    project_name: str
    item_id: int
    usage_statement_id: int
    report_month: str
    revision_no: int
    category_code: str
    category_name: str
    used_on: str
    item_name: str
    unit: str | None
    quantity: float
    unit_price: float
    total_amount: int
    remark: str | None
    page_no: int


@dataclass(slots=True)
class EvidenceType:
    """`service.evidence_types` 행을 정규화한 형태."""

    code: str
    name: str
    description: str


@dataclass(slots=True)
class EvidenceRequirement:
    """`service.evidence_requirements`의 active requirement 표현."""

    id: int | None
    usage_statement_item_id: int
    evidence_type_code: str
    is_satisfied: bool
    is_active: bool = True


@dataclass(slots=True)
class EvidenceFileLink:
    """`service.evidence_file_links`에서 읽은 증빙 연결 정보."""

    usage_statement_item_id: int
    file_id: int
    evidence_type_code: str


@dataclass(slots=True)
class LinkedEvidenceFileContext:
    """AI 입력용으로 평탄화한 항목별 연결 파일 문맥 뷰 응답."""

    item_id: int
    file_id: int
    original_filename: str
    mime_type: str
    uploaded_evidence_type_code: str
    linked_evidence_type_code: str
    storage_key: str
    captured_at: str | None
    uploaded_at: str | None


@dataclass(slots=True)
class AIEvidenceRequirementInput:
    """상세 항목 1건에 대해 LLM으로 보내는 입력 구조."""

    item_context: EvidenceRequirementItemContext
    linked_files: list[LinkedEvidenceFileContext]
    available_evidence_types: list[str]
    evidence_type_definitions: list[EvidenceType]


@dataclass(slots=True)
class AIEvidenceRequirementOutput:
    """필수 증빙 선택 결과를 담는 구조화된 LLM 출력."""

    required_evidences: list[str]
    confidence: float | None = None
    reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(slots=True)
class EvidenceStatusDetail:
    """증빙 코드별 제출 여부 비교 결과."""

    evidence_type_code: str
    required: bool
    submitted: bool


@dataclass(slots=True)
class EvidenceStatusResult:
    """상세 항목 1건에 대한 최종 증빙 상태 결과."""

    item_id: int
    status: str
    missing_evidences: list[str] = field(default_factory=list)
    submitted_evidences: list[str] = field(default_factory=list)
    details: list[EvidenceStatusDetail] = field(default_factory=list)
