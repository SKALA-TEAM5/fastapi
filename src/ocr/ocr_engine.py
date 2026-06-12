"""
OCR 엔진 통합 디스패치 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — ocr_engine.py

[역할]
  OCR_ENGINE 환경변수에 따라 CLOVA OCR 또는 VLM을 선택해 호출한다.
  라우터·파이프라인은 이 모듈만 호출하며 엔진 종류를 알 필요가 없다.

[전환 방법 — 코드 변경 없이 .env 한 줄만 수정]
  OCR_ENGINE=clova   → NAVER CLOVA OCR (인쇄 영수증, 저비용, 빠름)
  OCR_ENGINE=vlm     → Gemini / OpenAI VLM (수기 포함, 고품질) ← 기본값

[함수 목록]
  parse_receipt(file_path)              — 영수증 이미지 파싱
  parse_document_image(file_path, hint) — 거래명세표·임금명세서·세금계산서 이미지 파싱
  get_engine_name()                     — 현재 엔진명 반환 (로깅·헤더 등 활용)
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.config import (
    OCR_ENGINE,
    CLOVA_OCR_SECRET, CLOVA_OCR_URL,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════

def get_engine_name() -> str:
    """현재 활성 OCR 엔진명 반환."""
    return OCR_ENGINE  # "clova" | "vlm"


def parse_receipt(
    file_path: str,
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    """
    영수증 이미지 파싱.

    OCR_ENGINE=clova → CLOVA OCR 영수증 특화 모델
    OCR_ENGINE=vlm   → vlm_ocr.parse_vision_response (type_hint="receipt")

    Args:
        project_id:         토큰 사용량 기록용 프로젝트 ID (VLM 전용, 선택)
        usage_statement_id: 토큰 사용량 기록용 사용내역서 ID (VLM 전용, 선택)

    Returns:
        parse_clova_response() / parse_vision_response() 와 동일한 스키마
    """
    logger.debug("parse_receipt engine=%s file=%s", OCR_ENGINE, Path(file_path).name)

    if OCR_ENGINE == "clova":
        return _clova_parse_receipt(file_path)
    return _vlm_parse_receipt(file_path, project_id=project_id, usage_statement_id=usage_statement_id)


def parse_document_image(
    file_path: str,
    type_hint: str = "receipt",
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    """
    거래명세표 / 임금명세서 / 세금계산서 이미지 파싱.

    OCR_ENGINE=clova → CLOVA OCR + 정규식 파싱 (parse_tax_invoice.parse_from_image)
    OCR_ENGINE=vlm   → vlm_ocr.parse_vision_response

    Args:
        type_hint: VLM에 전달할 문서 유형 힌트
                   ("tax_invoice" | "receipt" | "transaction_statement")
        project_id:         토큰 사용량 기록용 프로젝트 ID (VLM 전용, 선택)
        usage_statement_id: 토큰 사용량 기록용 사용내역서 ID (VLM 전용, 선택)
    """
    logger.debug("parse_document_image engine=%s hint=%s file=%s",
                 OCR_ENGINE, type_hint, Path(file_path).name)

    if OCR_ENGINE == "clova":
        return _clova_parse_document(file_path)
    return _vlm_parse_document(file_path, type_hint, project_id=project_id, usage_statement_id=usage_statement_id)


# ══════════════════════════════════════════════
# 내부 구현 — CLOVA
# ══════════════════════════════════════════════

def _clova_parse_receipt(file_path: str) -> dict:
    from src.ocr.clova_ocr_receipt import (
        call_clova_receipt,
        parse_clova_response,
        validate_result,
    )
    raw    = call_clova_receipt(file_path, CLOVA_OCR_SECRET, CLOVA_OCR_URL)
    parsed = parse_clova_response(raw)
    parsed["source_file"] = Path(file_path).name
    return validate_result(parsed)


def _clova_parse_document(file_path: str) -> dict:
    from src.ocr.parse_tax_invoice import parse_from_image
    return parse_from_image(file_path, CLOVA_OCR_SECRET, CLOVA_OCR_URL)


# ══════════════════════════════════════════════
# 내부 구현 — VLM
# ══════════════════════════════════════════════

def _vlm_parse_receipt(
    file_path: str,
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    from src.ocr.vlm_ocr import parse_vision_response
    from src.ocr.receipt_validator import validate_result
    parsed = parse_vision_response(
        file_path, type_hint="receipt",
        project_id=project_id, usage_statement_id=usage_statement_id,
    )
    parsed["source_file"] = Path(file_path).name
    return validate_result(parsed)


def _vlm_parse_document(
    file_path: str,
    type_hint: str,
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    from src.ocr.vlm_ocr import parse_vision_response
    return parse_vision_response(
        file_path, type_hint=type_hint,
        project_id=project_id, usage_statement_id=usage_statement_id,
    )
