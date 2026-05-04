from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CATEGORIES: dict[str, str] = {
    "CAT_01": "안전관리자 등의 인건비 및 각종 업무 수당 등",
    "CAT_02": "안전시설비 등",
    "CAT_03": "보호구 등",
    "CAT_04": "안전보건진단비 등",
    "CAT_05": "안전보건교육비 등",
    "CAT_06": "근로자 건강장해예방비 등",
    "CAT_07": "건설재해예방 기술지도비",
    "CAT_08": "본사 안전전담부서 운영비",
    "CAT_09": "위험성평가 등에 따른 소요비용",
}

UNCLASSIFIED = "미분류"


class _KoreanAliasModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class DocumentClassification(BaseModel):
    """내부 분류 결과."""

    category_id: str = Field(description="분류된 카테고리 ID")
    category_name: str = Field(description="분류된 카테고리 한글명")
    confidence: float = Field(description="분류 확신도 (0.0~1.0)")
    total_amount: float = Field(description="증빙자료 총 금액 (원)")
    items: dict[str, float] = Field(description="입력 항목:금액 (참조용)")
    needs_human_review: bool = Field(default=False, description="사람 검토 필요 여부")
    review_reason: str = Field(default="", description="사람 검토 필요 사유")


class UsageStatementRow(_KoreanAliasModel):
    row_id: int | None = Field(default=None, alias="행ID")
    given_category_code: str = Field(alias="기존카테고리코드")
    used_on: str | None = Field(default=None, alias="사용일자")
    item_name: str = Field(alias="항목명")
    unit: str | None = Field(default=None, alias="단위")
    quantity: float | None = Field(default=None, alias="수량")
    unit_price: float | None = Field(default=None, alias="단가")
    total_amount: float = Field(alias="금액")


class UsageStatementReviewRequest(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    rows: list[UsageStatementRow] = Field(alias="항목목록")
    basic_info: dict[str, Any] = Field(default_factory=dict, alias="기본정보")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 1001,
                "기본정보": {},
                "항목목록": [
                    {
                        "행ID": 1,
                        "기존카테고리코드": "CAT_02",
                        "사용일자": "2026-04-30",
                        "항목명": "복합가스측정기",
                        "단위": "대",
                        "수량": 1,
                        "단가": 850000,
                        "금액": 850000,
                    },
                    {
                        "행ID": 2,
                        "기존카테고리코드": "CAT_02",
                        "사용일자": "2026-04-30",
                        "항목명": "안전타포린",
                        "단위": "EA",
                        "수량": 3,
                        "단가": 100000,
                        "금액": 300000,
                    },
                ],
            }
        },
    )


class RowReviewResult(_KoreanAliasModel):
    row_id: int | None = Field(default=None, alias="행ID")
    item_name: str = Field(alias="항목명")
    given_category_code: str = Field(alias="기존카테고리코드")
    final_category_code: str = Field(alias="최종카테고리코드")
    decision_status: Literal["유지", "카테고리변경", "검토필요"] = Field(alias="판정상태")
    needs_human_review: bool = Field(alias="검토필요여부")
    reason: str = Field(default="", alias="사유")


class UsageStatementReviewResponse(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    results: list[RowReviewResult] = Field(alias="검토결과")


class ClassifiedUsageStatementRow(_KoreanAliasModel):
    row_id: int | None = Field(default=None, alias="행ID")
    usage_statement_id: int | str | None = Field(default=None, alias="사용내역서ID")
    given_category_code: str = Field(alias="기존카테고리코드")
    used_on: str | None = Field(default=None, alias="사용일자")
    item_name: str = Field(alias="항목명")
    unit: str | None = Field(default=None, alias="단위")
    quantity: float | None = Field(default=None, alias="수량")
    unit_price: float | None = Field(default=None, alias="단가")
    total_amount: float = Field(alias="금액")
    final_category_code: str = Field(alias="최종카테고리코드")
    decision_status: Literal["유지", "카테고리변경", "검토필요"] = Field(alias="판정상태")
    needs_human_review: bool = Field(alias="검토필요여부")
    reason: str = Field(default="", alias="사유")


class UsageStatementItemsResponse(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    rows: list[ClassifiedUsageStatementRow] = Field(alias="항목목록")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 1001,
                "항목목록": [
                    {
                        "행ID": 1,
                        "사용내역서ID": 1001,
                        "기존카테고리코드": "CAT_02",
                        "사용일자": "2026-04-30",
                        "항목명": "복합가스측정기",
                        "단위": "대",
                        "수량": 1,
                        "단가": 850000,
                        "금액": 850000,
                        "최종카테고리코드": "CAT_04",
                        "판정상태": "카테고리변경",
                        "검토필요여부": False,
                        "사유": "",
                    },
                    {
                        "행ID": 2,
                        "사용내역서ID": 1001,
                        "기존카테고리코드": "CAT_02",
                        "사용일자": "2026-04-30",
                        "항목명": "안전타포린",
                        "단위": "EA",
                        "수량": 3,
                        "단가": 100000,
                        "금액": 300000,
                        "최종카테고리코드": "CAT_02",
                        "판정상태": "유지",
                        "검토필요여부": False,
                        "사유": "",
                    },
                ],
            }
        },
    )


class UsageStatementSingleItemRequest(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    row: UsageStatementRow = Field(alias="항목")
    basic_info: dict[str, Any] = Field(default_factory=dict, alias="기본정보")

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={
            "example": {
                "사용내역서ID": 1001,
                "기본정보": {},
                "항목": {
                    "행ID": 1,
                    "기존카테고리코드": "CAT_02",
                    "사용일자": "2026-04-30",
                    "항목명": "복합가스측정기",
                    "단위": "대",
                    "수량": 1,
                    "단가": 850000,
                    "금액": 850000,
                },
            }
        },
    )
