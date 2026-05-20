# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 상수 정의 ]
#
# 1. CATEGORIES                    : 카테고리 코드 → 명칭 매핑 딕셔너리
# 2. DocumentClassification        : 문서 분류 결과 스키마
# 3. UsageStatementRow             : 사용내역서 단일 행 스키마
# 4. UsageStatementReviewRequest   : 카테고리 분류 요청 스키마
# 5. UsageStatementReviewResponse  : 카테고리 분류 응답 스키마
# --------------------------------------------------------------------------
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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


class UnifiedUsageStatementRow(_KoreanAliasModel):
    """
    사용내역서 OCR 결과와 classifier 결과를 함께 담을 수 있는 공통 DTO.

    - 입력은 OCR 영문 키(`category_code`, `total_amount`, `line_no` 등)와
      classifier 한글 alias(`기존카테고리코드`, `금액`, `행ID` 등)를 모두 허용한다.
    - 출력은 기존 classifier 응답과의 호환을 위해 한글 alias 기준으로 직렬화한다.
    """

    row_id: int | None = Field(
        default=None,
        alias="행ID",
        validation_alias=AliasChoices("행ID", "row_id", "line_no"),
    )
    usage_statement_id: int | str | None = Field(
        default=None,
        alias="사용내역서ID",
        validation_alias=AliasChoices("사용내역서ID", "usage_statement_id"),
    )
    source_line_id: str | None = Field(
        default=None,
        alias="라인ID",
        validation_alias=AliasChoices("라인ID", "source_line_id", "line_id"),
    )
    given_category_code: str | None = Field(
        default=None,
        alias="기존카테고리코드",
        validation_alias=AliasChoices(
            "기존카테고리코드", "given_category_code", "category_code"
        ),
    )
    used_on: str | None = Field(
        default=None,
        alias="사용일자",
        validation_alias=AliasChoices("사용일자", "used_on"),
    )
    item_name: str | None = Field(
        default=None,
        alias="항목명",
        validation_alias=AliasChoices("항목명", "item_name"),
    )
    unit: str | None = Field(
        default=None,
        alias="단위",
        validation_alias=AliasChoices("단위", "unit"),
    )
    quantity: float | None = Field(
        default=None,
        alias="수량",
        validation_alias=AliasChoices("수량", "quantity"),
    )
    unit_price: float | None = Field(
        default=None,
        alias="단가",
        validation_alias=AliasChoices("단가", "unit_price"),
    )
    total_amount: float | None = Field(
        default=None,
        alias="금액",
        validation_alias=AliasChoices("금액", "total_amount", "amount"),
    )
    remark: str | None = Field(
        default=None,
        alias="비고",
        validation_alias=AliasChoices("비고", "remark"),
    )
    page_no: int | None = Field(
        default=None,
        alias="페이지번호",
        validation_alias=AliasChoices("페이지번호", "page_no"),
    )
    source_line_no: int | None = Field(
        default=None,
        alias="원본행번호",
        validation_alias=AliasChoices("원본행번호", "source_line_no", "line_no"),
    )
    final_category_code: str | None = Field(
        default=None,
        alias="최종카테고리코드",
        validation_alias=AliasChoices("최종카테고리코드", "final_category_code"),
    )
    decision_status: Literal["유지", "카테고리변경"] | None = Field(
        default=None,
        alias="판정상태",
        validation_alias=AliasChoices("판정상태", "decision_status"),
    )
    needs_human_review: bool | None = Field(
        default=None,
        alias="검토필요여부",
        validation_alias=AliasChoices("검토필요여부", "needs_human_review"),
    )
    reason: str = Field(
        default="",
        alias="사유",
        validation_alias=AliasChoices("사유", "reason"),
    )


class UsageStatementRow(_KoreanAliasModel):
    row_id: int | None = Field(
        default=None,
        alias="행ID",
        validation_alias=AliasChoices("행ID", "row_id", "line_no"),
    )
    given_category_code: str = Field(
        alias="기존카테고리코드",
        validation_alias=AliasChoices(
            "기존카테고리코드", "given_category_code", "category_code"
        ),
    )
    used_on: str | None = Field(
        default=None,
        alias="사용일자",
        validation_alias=AliasChoices("사용일자", "used_on"),
    )
    item_name: str = Field(
        alias="항목명",
        validation_alias=AliasChoices("항목명", "item_name"),
    )
    unit: str | None = Field(
        default=None,
        alias="단위",
        validation_alias=AliasChoices("단위", "unit"),
    )
    quantity: float | None = Field(
        default=None,
        alias="수량",
        validation_alias=AliasChoices("수량", "quantity"),
    )
    unit_price: float | None = Field(
        default=None,
        alias="단가",
        validation_alias=AliasChoices("단가", "unit_price"),
    )
    total_amount: float = Field(
        default=0,
        alias="금액",
        validation_alias=AliasChoices("금액", "total_amount", "amount"),
    )


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
    decision_status: Literal["유지", "카테고리변경"] = Field(alias="판정상태")
    needs_human_review: bool = Field(alias="검토필요여부")
    reason: str = Field(default="", alias="사유")


class UsageStatementReviewResponse(_KoreanAliasModel):
    usage_statement_id: int | str = Field(alias="사용내역서ID")
    results: list[RowReviewResult] = Field(alias="검토결과")


class ClassifiedUsageStatementRow(_KoreanAliasModel):
    row_id: int | None = Field(
        default=None,
        alias="행ID",
        validation_alias=AliasChoices("행ID", "row_id", "line_no"),
    )
    usage_statement_id: int | str | None = Field(
        default=None,
        alias="사용내역서ID",
        validation_alias=AliasChoices("사용내역서ID", "usage_statement_id"),
    )
    source_line_id: str | None = Field(
        default=None,
        alias="라인ID",
        validation_alias=AliasChoices("라인ID", "source_line_id", "line_id"),
    )
    given_category_code: str = Field(
        alias="기존카테고리코드",
        validation_alias=AliasChoices(
            "기존카테고리코드", "given_category_code", "category_code"
        ),
    )
    used_on: str | None = Field(
        default=None,
        alias="사용일자",
        validation_alias=AliasChoices("사용일자", "used_on"),
    )
    item_name: str = Field(
        alias="항목명",
        validation_alias=AliasChoices("항목명", "item_name"),
    )
    unit: str | None = Field(
        default=None,
        alias="단위",
        validation_alias=AliasChoices("단위", "unit"),
    )
    quantity: float | None = Field(
        default=None,
        alias="수량",
        validation_alias=AliasChoices("수량", "quantity"),
    )
    unit_price: float | None = Field(
        default=None,
        alias="단가",
        validation_alias=AliasChoices("단가", "unit_price"),
    )
    total_amount: float = Field(
        alias="금액",
        validation_alias=AliasChoices("금액", "total_amount", "amount"),
    )
    remark: str | None = Field(
        default=None,
        alias="비고",
        validation_alias=AliasChoices("비고", "remark"),
    )
    page_no: int | None = Field(
        default=None,
        alias="페이지번호",
        validation_alias=AliasChoices("페이지번호", "page_no"),
    )
    source_line_no: int | None = Field(
        default=None,
        alias="원본행번호",
        validation_alias=AliasChoices("원본행번호", "source_line_no", "line_no"),
    )
    final_category_code: str = Field(
        alias="최종카테고리코드",
        validation_alias=AliasChoices("최종카테고리코드", "final_category_code"),
    )
    decision_status: Literal["유지", "카테고리변경"] = Field(
        alias="판정상태",
        validation_alias=AliasChoices("판정상태", "decision_status"),
    )
    needs_human_review: bool = Field(
        alias="검토필요여부",
        validation_alias=AliasChoices("검토필요여부", "needs_human_review"),
    )
    reason: str = Field(
        default="",
        alias="사유",
        validation_alias=AliasChoices("사유", "reason"),
    )


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
