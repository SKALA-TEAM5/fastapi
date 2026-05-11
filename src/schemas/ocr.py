"""
산업안전보건관리비 OCR 검증 API — Pydantic 스키마
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI 요청/응답 모델 정의.
Swagger UI에서 자동으로 문서화된다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════
# 단일 파싱 엔드포인트 (POST /ocr/parse)
# ══════════════════════════════════════════════

class FileRecord(BaseModel):
    """
    POST /ocr/parse 요청 — DB files 테이블 레코드를 그대로 전달
    OCR 엔진은 storage_key 로 S3 에서 파일을 직접 가져온다.
    """
    id: int = Field(..., description="files PK (BIGSERIAL)", examples=[1])
    project_id: Optional[int] = Field(None, description="프로젝트 ID (미사용)", examples=[10])
    uploaded_by_user_id: Optional[int] = Field(None, description="업로드 사용자 ID (미사용)", examples=[5])
    uploaded_evidence_type_code: str = Field(
        ...,
        description=(
            "파서 분기 기준:\n"
            "- `usage_statement` : 사용내역서 — 파싱만, 매칭 없음\n"
            "- `receipt` : 영수증 — CLOVA OCR\n"
            "- `tax_invoice` : 전자세금계산서 — pdfplumber / CLOVA OCR\n"
            "- `transaction_statement` : 거래명세표 — pdfplumber / 정규식\n"
            "- `wage_statement` : 임금명세서 — 거래명세표 파서 공유, 매칭 없음"
        ),
        examples=["receipt"],
    )
    original_filename: str = Field(..., description="원본 파일명", examples=["안전화_영수증_20260422.jpg"])
    storage_key: str = Field(..., description="S3 오브젝트 키", examples=["projects/10/receipts/안전화_영수증_20260422.jpg"])
    mime_type: str = Field(..., description="PDF/이미지 파서 선택에 사용", examples=["image/jpeg"])
    size_bytes: Optional[int] = Field(None, description="파일 크기 (미사용)", examples=[204800])
    captured_at: Optional[datetime] = Field(None, description="촬영/작성 일시 (선택)")
    uploaded_at: Optional[datetime] = Field(None, description="업로드 일시")


class ParseError(BaseModel):
    """비즈니스 실패 에러 상세"""
    code: str = Field(..., description="에러 코드", examples=["unmatched"])
    message: str = Field(..., description="에러 메시지", examples=["매칭 실패: 유사도 0.52"])


class ParseResponse(BaseModel):
    """
    POST /ocr/parse 공통 응답 래퍼
    - 성공: success=true, data에 파싱/매칭 결과
    - 비즈니스 실패: success=false, error에 코드와 메시지
    """
    success: bool = Field(..., description="처리 성공 여부")
    data: Optional[Any] = Field(None, description="파싱 결과 (성공 시)")
    error: Optional[ParseError] = Field(None, description="에러 상세 (실패 시)")
    message: str = Field("", description="처리 결과 메시지")


# ══════════════════════════════════════════════
# 영수증 OCR
# ══════════════════════════════════════════════

class ReceiptItem(BaseModel):
    """영수증 내 개별 품목"""
    item_name: str = Field(..., description="품목명", examples=["안전화"])
    count: Optional[int] = Field(None, description="수량", examples=[5])
    unit_price: Optional[int] = Field(None, description="단가 (원)", examples=[30000])
    amount: Optional[int] = Field(None, description="금액 (원) — OCR 미인식 시 null", examples=[150000])
    roi_box: Optional[list[int]] = Field(
        None,
        description="영수증 내 위치 좌표 [x, y, w, h] — 프론트 하이라이팅용",
        examples=[[120, 340, 280, 60]],
    )


class ReceiptValidation(BaseModel):
    """영수증 자체 유효성 검증 결과"""
    is_valid: bool = Field(..., description="영수증 유효 여부")
    items_sum_match: Optional[bool] = Field(None, description="품목 합계 = 총액 여부 (품목 없으면 null)")
    warnings: list[str] = Field(default_factory=list, description="경고/오류 메시지 목록")


class ReceiptOCRResponse(BaseModel):
    """POST /receipts/ocr 응답 — 영수증·거래명세표·임금명세서 공통 구조"""
    receipt_id: str = Field(..., description="문서 고유 ID (UUID)", examples=["rec_20250415_001"])
    source_file: str = Field(..., description="원본 파일명", examples=["보호구_거래명세표_20250415.jpg"])
    doc_type: str = Field(
        ...,
        description=(
            "문서 유형 (파일명·PDF 텍스트 키워드로 자동 판별):\n"
            "- `receipt` : 영수증 — CLOVA 영수증 모델 파싱, 카드 승인일시 기준\n"
            "- `transaction_statement` : 거래명세표 — pdfplumber/정규식 파싱, 작성일자 기준\n"
            "- `wage_statement` : 임금명세서 — 거래명세표와 동일 파서, Gate 3·세금계산서 검증 면제"
        ),
        examples=["receipt"],
    )
    infer_result: str = Field(..., description="OCR 처리 결과 코드 (SUCCESS | PARTIAL | FAILURE | ERROR)", examples=["SUCCESS"])
    vendor: Optional[str] = Field(
        None,
        description="거래처명 (영수증: 카드 매장명 / 거래명세표: 공급자 상호 / 임금명세서: null 가능)",
        examples=["한국안전용품"],
    )
    date: Optional[str] = Field(
        None,
        description="거래 날짜 YYYY-MM-DD (영수증: 카드 승인일시 / 거래명세표·임금명세서: 작성일자)",
        examples=["2025-04-15"],
    )
    total_amount: Optional[int] = Field(None, description="합계 금액 (원)", examples=[300000])
    items: list[ReceiptItem] = Field(default_factory=list, description="품목 목록")
    validation: ReceiptValidation


# ══════════════════════════════════════════════
# 매칭
# ══════════════════════════════════════════════

class UsageItem(BaseModel):
    """사용내역서 단일 항목 (매칭 요청 시 입력)"""
    seq: int = Field(..., description="항목 순번", examples=[1])
    category_code: str = Field(..., description="안전관리비 항목 코드", examples=["CAT_03"])
    used_on: str = Field(..., description="사용 일자 (YYYY-MM-DD)", examples=["2025-04-15"])
    item_name: str = Field(..., description="품목명", examples=["안전모"])
    total_amount: int = Field(..., description="합계 금액 (원)", examples=[150000])
    remark: Optional[str] = Field(None, description="비고")


class MatchRequest(BaseModel):
    """POST /matching/run 요청"""
    project_id: int = Field(..., description="프로젝트 ID", examples=[1])
    usage_statement_id: int = Field(..., description="사용내역서 DB ID", examples=[1])
    usage_items: list[UsageItem] = Field(..., description="사용내역서 항목 목록")
    receipt_ocr_results: list[dict] = Field(
        ...,
        description="clova_ocr_receipt 파싱 결과 목록 (ReceiptOCRResponse 구조)",
    )
    photo_texts: Optional[dict[int, str]] = Field(
        None,
        description="현장사진 텍스트 {seq: OCR 텍스트} (선택)",
    )
    save_to_db: bool = Field(
        True,
        description="매칭 결과를 validation_logs에 저장할지 여부",
    )


class ComponentScores(BaseModel):
    """매칭 세부 점수"""
    date: Optional[float] = Field(None, description="날짜 일치 점수 (0~1)")
    amount: Optional[float] = Field(None, description="금액 일치 점수 (0~1)")
    vendor: Optional[float] = Field(None, description="업체명 유사도 (0~1)")
    item_desc: Optional[float] = Field(None, description="품목명 유사도 (0~1)")


class MatchResultItem(BaseModel):
    """단일 항목의 매칭 결과"""
    usage_item: UsageItem
    receipt: Optional[dict] = Field(None, description="매칭된 영수증 요약 (없으면 null)")
    match_status: str = Field(
        ...,
        description="매칭 상태",
        examples=["matched"],
    )
    similarity_score: float = Field(..., description="종합 유사도 점수 (0~1)", examples=[0.97])
    component_scores: ComponentScores
    gate_failed: list[str] = Field(
        default_factory=list,
        description="실패한 Gate 목록",
        examples=[["amount_gate"]],
    )
    reject_reason: Optional[str] = Field(None, description="rejected 사유")


class MatchSummary(BaseModel):
    """배치 매칭 요약"""
    total: int
    matched: int
    review_needed: int
    unmatched: int
    rejected: int
    match_rate_pct: float
    review_rate_pct: float


class MatchResponse(BaseModel):
    """POST /matching/run 응답"""
    batch_id: str = Field(..., description="배치 고유 ID (UUID)")
    created_at: str = Field(..., description="실행 일시")
    thresholds: dict = Field(..., description="사용된 임계값 {matched, review}")
    summary: MatchSummary
    results: list[MatchResultItem]


# ══════════════════════════════════════════════
# 세금계산서 OCR
# ══════════════════════════════════════════════

class TaxInvoiceParty(BaseModel):
    """공급자 / 공급받는자 정보"""
    company_name:    Optional[str] = Field(None, description="상호", examples=["한국안전용품(주)"])
    business_number: Optional[str] = Field(None, description="사업자등록번호 (XXX-XX-XXXXX)", examples=["105-87-46831"])
    representative:  Optional[str] = Field(None, description="대표자 성명", examples=["홍길동"])


class TaxInvoiceItem(BaseModel):
    """세금계산서 품목 단일 행"""
    name:          Optional[str] = Field(None, description="품목명", examples=["안전모"])
    spec:          Optional[str] = Field(None, description="규격", examples=["KCS 표준형"])
    count:         Optional[int] = Field(None, description="수량", examples=[10])
    unit_price:    Optional[int] = Field(None, description="단가 (원)", examples=[15000])
    supply_amount: Optional[int] = Field(None, description="공급가액 (원)", examples=[150000])
    tax_amount:    Optional[int] = Field(None, description="세액 (원)", examples=[15000])


class TaxInvoiceValidation(BaseModel):
    """세금계산서 파싱 후 검증 결과"""
    is_valid: bool = Field(..., description="필수 필드 및 금액 검증 통과 여부")
    warnings: list[str] = Field(default_factory=list, description="경고/오류 메시지 목록")


class TaxInvoiceOCRResponse(BaseModel):
    """POST /tax-invoices/ocr 응답"""
    invoice_id:          str = Field(..., description="세금계산서 고유 ID (UUID)", examples=["inv_3a7f9c12"])
    source_file:         str = Field(..., description="원본 파일명", examples=["세금계산서_20251005.pdf"])
    parse_method:        str = Field(..., description="파싱 방식 (pdfplumber | ocr)", examples=["pdfplumber"])
    supplier:            TaxInvoiceParty = Field(..., description="공급자 정보")
    buyer:               TaxInvoiceParty = Field(..., description="공급받는자 정보")
    issue_date:          Optional[str] = Field(None, description="작성일자 (YYYY-MM-DD)", examples=["2025-10-05"])
    items:               list[TaxInvoiceItem] = Field(default_factory=list, description="품목 목록")
    total_supply_amount: Optional[int] = Field(None, description="공급가액 합계 (원)", examples=[300000])
    total_tax_amount:    Optional[int] = Field(None, description="세액 합계 (원)", examples=[30000])
    total_amount:        Optional[int] = Field(None, description="합계금액 (원)", examples=[330000])
    validation:          TaxInvoiceValidation


# ══════════════════════════════════════════════
# 검증 로그 & HITL
# ══════════════════════════════════════════════

class ValidationLogItem(BaseModel):
    """validation_logs 단일 행"""
    id: int
    usage_statement_item_id: Optional[int] = Field(None, description="사용내역서 항목 DB ID")
    item_name: Optional[str] = Field(None, description="품목명 (JOIN 결과)")
    total_amount: Optional[int] = Field(None, description="신고 금액 (원)")
    validation_type_code: str = Field(..., description="검증 유형", examples=["ocr_receipt_match"])
    result_code: str = Field(
        ...,
        description="결과 코드",
        examples=["matched"],
    )
    gate_failed: Optional[list[str]] = Field(None, description="실패 Gate 목록")
    similarity_score: Optional[float] = Field(None, description="유사도 점수")
    model_name: Optional[str] = Field(None, description="사용 모델명")
    created_at: datetime


class ReviewRequest(BaseModel):
    """POST /validation-logs/{log_id}/review 요청"""
    decision: str = Field(
        ...,
        description="검토 결정",
        examples=["approved"],
        pattern="^(approved|rejected)$",
    )
    reason: str = Field("", description="결정 사유 (선택)", examples=["현장 확인 후 적합 판단"])
    reviewer_id: int = Field(..., description="검토자 ID", examples=[1])


class ReviewResponse(BaseModel):
    """POST /validation-logs/{log_id}/review 응답"""
    log_id: int = Field(..., description="생성된 human_review 로그 ID")
    original_log_id: int = Field(..., description="검토 대상 ocr_receipt_match 로그 ID")
    decision: str
    reviewer_id: int
    created_at: datetime
