from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Audit engine models ──────────────────────────────────────────────

CategoryStatus = Literal["적절", "부적절", "검토필요"]


class _ItemJudgeLLMOutput(BaseModel):
    """LLM이 채우는 항목 판정 결과. category는 입력에서 이미 알고 있으므로 LLM 미결정."""

    allowed: bool = Field(description="해당 항목 산안비 집행 허용 여부")
    confidence: float = Field(description="판정 확신도 (0.0~1.0)")
    reasoning: str = Field(description="판정 근거 (법령 조항 인용 포함)")
    evidence_snippets: List[str] = Field(
        default=[], description="Context 원문 발췌 1~3개"
    )
    referenced_laws: List[str] = Field(default=[], description="참조 법령 조항 목록")
    category_limit_pct: Optional[float] = Field(
        default=None,
        description="이 카테고리의 법령상 총액 한도 비율 (0~1 소수, 예: 20% → 0.2). 명시적 근거 없으면 null.",
    )
    category_limit_rule: str = Field(
        default="",
        description="한도 규정 원문 발췌 (없으면 빈 문자열)",
    )


class ItemJudgment(BaseModel):
    item: str = Field(description="항목명")
    amount: float = Field(description="집행액 (원)")
    category: str = Field(description="소속 법정 카테고리 (입력값)")
    allowed: bool = Field(description="해당 항목 산안비 집행 허용 여부")
    confidence: float = Field(description="판정 확신도 (0.0~1.0)")
    reasoning: str = Field(description="판정 근거")
    evidence_snippets: List[str] = Field(default=[])
    referenced_laws: List[str] = Field(default=[])
    category_limit_pct: Optional[float] = Field(default=None)
    category_limit_rule: str = Field(default="")
    needs_human_review: bool = Field(default=False)
    review_reason: str = Field(default="")
    exception_summary: str = Field(default="")


class CategoryAuditResult(BaseModel):
    status: CategoryStatus = Field(description="적절 | 부적절 | 검토필요")
    total: float = Field(description="카테고리 집행 합계 (원)")
    limit: Optional[float] = Field(default=None, description="법령상 한도액 (원)")
    exceeded: bool = Field(description="한도 초과 여부")
    limit_rule: str = Field(default="", description="한도 규정 원문 (RAG 추출)")
    rejection_reason: str = Field(default="", description="부적절/검토필요 사유")
    llm_interpretation: str = Field(
        default="", description="LLM이 청크 기반으로 생성한 판정 해석"
    )
    llm_improvements: str = Field(
        default="", description="LLM이 청크 기반으로 생성한 보완사항"
    )
    items: List[ItemJudgment] = Field(default=[])
    referenced_laws: List[str] = Field(default=[])
    evidence_snippets: List[str] = Field(default=[])
    needs_human_review: bool = Field(default=False)
    progress_rate: Optional[float] = Field(
        default=None, description="현재 누계 공정률 (%)"
    )
    required_usage_rate: Optional[float] = Field(
        default=None, description="공정률 구간상 요구 최소 사용률 (0~1)"
    )
    required_used_amount: Optional[float] = Field(
        default=None, description="공정률 기준 요구 최소 사용액 (원)"
    )
    cumulative_used_amount: Optional[float] = Field(
        default=None, description="실제 누적 사용액 (원)"
    )
    usage_shortfall_amount: Optional[float] = Field(
        default=None, description="공정률 기준 부족액 (원)"
    )


class AuditResponse(BaseModel):
    base_amount: float
    categories: dict[str, CategoryAuditResult]


class _KoreanAliasModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class AuditSourceSummary(_KoreanAliasModel):
    law: str = Field(alias="조항")
    summary: str = Field(alias="요지")


class CategoryAuditSummary(_KoreanAliasModel):
    category_code: str = Field(alias="카테고리코드")
    status: CategoryStatus = Field(alias="판정상태")
    reason: str = Field(alias="사유")
    sources: List[AuditSourceSummary] = Field(alias="출처")


class UsageStatementAuditSummaryResponse(_KoreanAliasModel):
    usage_statement_id: int | str | None = Field(default=None, alias="사용내역서ID")
    results: List[CategoryAuditSummary] = Field(alias="검토결과")


class ValidatorInputBasicInfo(_KoreanAliasModel):
    base_amount: float = Field(alias="산안비총액")
    progress_rate: float | None = Field(default=None, alias="누계공정률")


class ValidatorSummaryInput(_KoreanAliasModel):
    previous_amount: float = Field(alias="전회사용금액")
    current_amount: float = Field(alias="금회사용금액")
    cumulative_amount: float = Field(alias="누적사용금액")


class ValidatorItemInput(_KoreanAliasModel):
    row_id: int | None = Field(default=None, alias="행ID")
    used_on: str | None = Field(default=None, alias="사용일자")
    item_name: str = Field(alias="항목명")
    unit: str | None = Field(default=None, alias="단위")
    quantity: float | None = Field(default=None, alias="수량")
    unit_price: float | None = Field(default=None, alias="단가")
    total_amount: float = Field(alias="금액")
    remark: str = Field(default="", alias="비고")


class ValidatorCategoryInput(_KoreanAliasModel):
    category_code: str = Field(alias="카테고리코드")
    summary: ValidatorSummaryInput = Field(alias="집계정보")
    items: List[ValidatorItemInput] = Field(alias="항목목록")


class UsageStatementValidatorRequest(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    basic_info: ValidatorInputBasicInfo = Field(alias="기본정보")
    categories: List[ValidatorCategoryInput] = Field(alias="카테고리별데이터")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 2008,
                "기본정보": {
                    "산안비총액": 50000000,
                    "누계공정률": 72.5,
                },
                "카테고리별데이터": [
                    {
                        "카테고리코드": "CAT_02",
                        "집계정보": {
                            "전회사용금액": 21000000,
                            "금회사용금액": 1500000,
                            "누적사용금액": 22500000,
                        },
                        "항목목록": [
                            {
                                "행ID": 1,
                                "사용일자": "2026-04-30",
                                "항목명": "안전타포린(추락위험, 접근금지)",
                                "단위": "EA",
                                "수량": 10,
                                "단가": 150000,
                                "금액": 1500000,
                                "비고": "",
                            }
                        ],
                    }
                ],
            }
        },
    )


class UsageStatementValidatorCategoryRequest(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    basic_info: ValidatorInputBasicInfo = Field(alias="기본정보")
    category: ValidatorCategoryInput = Field(alias="카테고리데이터")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 2008,
                "기본정보": {
                    "산안비총액": 50000000,
                    "누계공정률": 72.5,
                },
                "카테고리데이터": {
                    "카테고리코드": "CAT_02",
                    "집계정보": {
                        "전회사용금액": 21000000,
                        "금회사용금액": 1500000,
                        "누적사용금액": 22500000,
                    },
                    "항목목록": [
                        {
                            "행ID": 1,
                            "사용일자": "2026-04-30",
                            "항목명": "안전타포린(추락위험, 접근금지)",
                            "단위": "EA",
                            "수량": 10,
                            "단가": 150000,
                            "금액": 1500000,
                            "비고": "",
                        }
                    ],
                },
            }
        },
    )


class UsageStatementSingleCategoryAuditResponse(_KoreanAliasModel):
    usage_statement_id: int | str | None = Field(default=None, alias="사용내역서ID")
    result: CategoryAuditSummary = Field(alias="검토결과")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 2008,
                "검토결과": {
                    "카테고리코드": "CAT_02",
                    "판정상태": "검토필요",
                    "사유": "제7조제1항제2호에 따르면 산업재해 예방을 위한 안전시설, 안전장비, 화재위험작업용 소화기 등은 안전시설비 항목에 해당합니다. 공정률 72.5% 구간에서는 총액의 70% 이상 사용해야 하나 실제 누적 사용액이 부족해 검토가 필요합니다. 보완점으로는 누적 집행 계획과 집행 시점을 다시 확인할 필요가 있습니다.",
                    "출처": [
                        {
                            "조항": "제7조제1항제2호",
                            "요지": "산업재해 예방을 위한 안전시설, 안전장비, 화재위험작업용 소화기 등은 안전시설비 항목에 해당합니다.",
                        },
                        {
                            "조항": "별표 3 공사진척에 따른 산업안전보건관리비 사용기준",
                            "요지": "공정률 70% 이상 90% 미만 구간에서는 산업안전보건관리비를 70% 이상 사용해야 합니다.",
                        },
                    ],
                },
            }
        },
    )


class ValidatorCategoryDataWithResult(_KoreanAliasModel):
    category_code: str = Field(alias="카테고리코드")
    summary: ValidatorSummaryInput = Field(alias="집계정보")
    items: List[ValidatorItemInput] = Field(alias="항목목록")
    result: CategoryAuditSummary = Field(alias="검토결과")


class UsageStatementValidatorEmbeddedResponse(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    basic_info: ValidatorInputBasicInfo = Field(alias="기본정보")
    categories: List[ValidatorCategoryDataWithResult] = Field(alias="카테고리별데이터")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 2008,
                "기본정보": {
                    "산안비총액": 50000000,
                    "누계공정률": 72.5,
                },
                "카테고리별데이터": [
                    {
                        "카테고리코드": "CAT_02",
                        "집계정보": {
                            "전회사용금액": 21000000,
                            "금회사용금액": 1500000,
                            "누적사용금액": 22500000,
                        },
                        "항목목록": [
                            {
                                "행ID": 1,
                                "사용일자": "2026-04-30",
                                "항목명": "안전타포린(추락위험, 접근금지)",
                                "단위": "EA",
                                "수량": 10,
                                "단가": 150000,
                                "금액": 1500000,
                                "비고": "",
                            }
                        ],
                        "검토결과": {
                            "카테고리코드": "CAT_02",
                            "판정상태": "검토필요",
                            "사유": "제7조제1항제2호에 따르면 산업재해 예방을 위한 안전시설과 안전장비는 안전시설비 항목에 해당합니다. 공정률 72.5% 구간에서는 총액의 70% 이상 사용해야 하나 실제 누적 사용액이 부족해 검토가 필요합니다. 보완점으로는 누적 집행 계획과 집행 시점을 다시 확인할 필요가 있습니다.",
                            "출처": [
                                {
                                    "조항": "제7조제1항제2호",
                                    "요지": "산업재해 예방을 위한 안전시설, 안전장비, 화재위험작업용 소화기 등은 안전시설비 항목에 해당합니다.",
                                },
                                {
                                    "조항": "별표 3 공사진척에 따른 산업안전보건관리비 사용기준",
                                    "요지": "공정률 70% 이상 90% 미만 구간에서는 산업안전보건관리비를 70% 이상 사용해야 합니다.",
                                },
                            ],
                        },
                    }
                ],
            }
        },
    )


class UsageStatementValidatorCategoryEmbeddedResponse(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    basic_info: ValidatorInputBasicInfo = Field(alias="기본정보")
    category: ValidatorCategoryDataWithResult = Field(alias="카테고리데이터")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 2008,
                "기본정보": {
                    "산안비총액": 50000000,
                    "누계공정률": 72.5,
                },
                "카테고리데이터": {
                    "카테고리코드": "CAT_02",
                    "집계정보": {
                        "전회사용금액": 21000000,
                        "금회사용금액": 1500000,
                        "누적사용금액": 22500000,
                    },
                    "항목목록": [
                        {
                            "행ID": 1,
                            "사용일자": "2026-04-30",
                            "항목명": "안전타포린(추락위험, 접근금지)",
                            "단위": "EA",
                            "수량": 10,
                            "단가": 150000,
                            "금액": 1500000,
                            "비고": "",
                        }
                    ],
                    "검토결과": {
                        "카테고리코드": "CAT_02",
                        "판정상태": "적절",
                        "사유": "제7조제1항제2호에 따르면 산업재해 예방을 위한 안전시설과 안전장비는 안전시설비 항목에 해당합니다. 해당 항목은 일반 허용 범위에 포함되고 수치 기준도 충족하여 적절합니다. 보완점으로는 관련 증빙을 함께 정리해 두면 후속 검토에 도움이 됩니다.",
                        "출처": [
                            {
                                "조항": "제7조제1항제2호",
                                "요지": "산업재해 예방을 위한 안전시설, 안전장비, 화재위험작업용 소화기 등은 안전시설비 항목에 해당합니다.",
                            }
                        ],
                    },
                },
            }
        },
    )


class ValidatorCategoryMetrics(BaseModel):
    confidence: float = Field(default=0.0, description="평균 판정 확신도")
    total: float = Field(description="카테고리 집행 합계 (원)")
    limit: Optional[float] = Field(default=None, description="법령상 한도액 (원)")
    exceeded: bool = Field(default=False, description="한도 초과 여부")
    needs_human_review: bool = Field(default=False)
    progress_rate: Optional[float] = Field(default=None)
    required_usage_rate: Optional[float] = Field(default=None)
    usage_shortfall_amount: Optional[float] = Field(default=None)


class ValidatorAuditResponse(BaseModel):
    result: UsageStatementAuditSummaryResponse
    metrics: dict[str, ValidatorCategoryMetrics]
