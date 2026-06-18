# --------------------------------------------------------------------------
# 작성자   : 이현수(kacalu0930)
# 작성일   : 2026-06-04
# 수정일   : 2026-06-18 (Clova→VLM 전환: OCR_ENGINE 분기·CLOVA 함수 제거, VLM 전용화)
#
# [ 주요 함수 정의 ]
#
# 1. get_engine_name()       : 현재 OCR 엔진명("vlm") 반환
# 2. parse_receipt()         : 영수증 이미지 VLM 파싱
# 3. parse_document_image()  : 거래명세표/임금명세서/세금계산서 이미지 VLM 파싱
# --------------------------------------------------------------------------
"""
OCR 엔진 통합 디스패치 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — ocr_engine.py

[역할]
  영수증·문서 이미지 OCR을 VLM(Gemini/OpenAI) 경로로 수행한다.
  라우터·파이프라인은 이 모듈만 호출하며 내부 구현을 알 필요가 없다.

  ※ 과거 OCR_ENGINE 환경변수로 CLOVA OCR / VLM을 전환했으나,
    VLM 전면 사용으로 전환하며 CLOVA 경로는 제거되었다.

[함수 목록]
  parse_receipt(file_path)              — 영수증 이미지 파싱
  parse_document_image(file_path, hint) — 거래명세표·임금명세서·세금계산서 이미지 파싱
  get_engine_name()                     — 현재 엔진명 반환 (로깅·헤더 등 활용)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# [Clova→VLM 리팩토링] 기존 OCR_ENGINE(clova/vlm) 분기·CLOVA 호출 함수를 전부 제거하고
#   VLM 전용 디스패처로 단순화. get_engine_name()은 항상 "vlm" 반환.
ENGINE_NAME = "vlm"


# ══════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════

def get_engine_name() -> str:
    """현재 활성 OCR 엔진명 반환."""
    return ENGINE_NAME


def parse_receipt(
    file_path: str,
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    """
    영수증 이미지 파싱 (VLM, type_hint="receipt").

    Args:
        project_id:         토큰 사용량 기록용 프로젝트 ID (선택)
        usage_statement_id: 토큰 사용량 기록용 사용내역서 ID (선택)

    Returns:
        parse_vision_response() 표준 스키마
    """
    logger.debug("parse_receipt engine=%s file=%s", ENGINE_NAME, Path(file_path).name)

    from src.ocr.vlm_ocr import parse_vision_response
    from src.ocr.receipt_validator import validate_result
    parsed = parse_vision_response(
        file_path, type_hint="receipt",
        project_id=project_id, usage_statement_id=usage_statement_id,
    )
    parsed["source_file"] = Path(file_path).name
    return validate_result(parsed)


def parse_document_image(
    file_path: str,
    type_hint: str = "receipt",
    project_id: int | None = None,
    usage_statement_id: int | None = None,
) -> dict:
    """
    거래명세표 / 임금명세서 / 세금계산서 이미지 파싱 (VLM).

    Args:
        type_hint: VLM에 전달할 문서 유형 힌트
                   ("tax_invoice" | "receipt" | "transaction_statement" | "wage_statement")
        project_id:         토큰 사용량 기록용 프로젝트 ID (선택)
        usage_statement_id: 토큰 사용량 기록용 사용내역서 ID (선택)
    """
    logger.debug("parse_document_image engine=%s hint=%s file=%s",
                 ENGINE_NAME, type_hint, Path(file_path).name)

    from src.ocr.vlm_ocr import parse_vision_response
    return parse_vision_response(
        file_path, type_hint=type_hint,
        project_id=project_id, usage_statement_id=usage_statement_id,
    )
