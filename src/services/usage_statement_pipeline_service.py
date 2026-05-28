"""
사용내역서 파이프라인 서비스 (API 호출용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용내역서 OCR 파싱, classifier 반영, 증빙 링크 후속 처리를
API에서 직접 호출 가능한 함수 형태로 제공한다.

제공 함수:
  - parse_usage_statement  : 사용내역서 PDF 파싱 → DB 저장 (/ocr/parse 용)
  - run_link_pipeline      : 영수증·세금계산서 OCR + 2-way 매칭 → DB 저장 (/link/run 용)

호출 시점:
  - /ocr/parse  : Spring이 파일 업로드 직후 즉시 호출 → usage_statement_id 반환
  - /link/run   : 분류 Agent + safety-doc 완료 후 Spring이 호출
                  (usage_statement_id + receipt/tax_invoice file_ids 전달)

로그 커넥션 설계:
  - insert_agent_log                   : 별도 커넥션으로 즉시 커밋 (메인 트랜잭션 롤백과 무관)
  - update_agent_log_status('completed'): 메인 커넥션 마지막에 호출
  - update_agent_log_status('failed')  : except 절에서 별도 커넥션으로 호출
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from src.core.config import CLOVA_OCR_SECRET, CLOVA_OCR_URL
from src.ocr.clova_ocr_receipt import (
    call_clova_receipt,
    parse_clova_response,
    validate_result as validate_ocr_result,
    SUPPORTED_EXTS,
)
from src.ocr.parse_tax_invoice import parse_tax_invoice, ALL_EXTS as TAX_INVOICE_EXTS
from src.ocr.parse_usage_statement import parse_pdf as parse_usage_pdf
from src.agents.classifier_agent.agent import review_usage_statement
from src.repositories.db import get_connection
from src.repositories.usage_statement_pipeline_repository import (
    get_files_by_ids,
    insert_evidence_file_link,
    insert_usage_statement,
    insert_usage_statement_items,
    insert_usage_statement_summaries,
    insert_agent_log,
    update_agent_log_status,
    update_file_status,
    _from_category_code,
)
from src.services.matching_service_monthly import (
    match_all_usage_to_receipts,
    THRESHOLD_MATCHED,
    THRESHOLD_REVIEW,
)
from src.services.s3_client import fetch_file


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────

def _fetch_and_save_temp(storage_key: str, suffix: str) -> str:
    """S3에서 파일을 가져와 임시 파일로 저장하고 경로를 반환한다."""
    file_bytes = fetch_file(storage_key)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name


def _cleanup(*paths: str) -> None:
    """임시 파일들을 삭제한다."""
    for p in paths:
        Path(p).unlink(missing_ok=True)


def _build_classifier_basic_info(parsed_usage: dict[str, Any]) -> dict[str, Any]:
    """사용내역서 header를 classifier 기본정보로 정리한다."""
    header = parsed_usage.get("header") or {}
    return {
        key: value
        for key, value in header.items()
        if key != "category_summaries" and value not in (None, "", [])
    }


def _classify_usage_statement(
    *,
    usage_file_id: int,
    parsed_usage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    OCR 파싱된 사용내역서를 classifier에 태워 카테고리를 보정하고,
    agent_logs.details에 저장할 classifier JSON도 함께 만든다.
    """
    line_items = parsed_usage.get("line_items") or []
    if not line_items:
        classifier_details = {
            "event": "classification_checked",
            "summary": "분류할 세부내역이 없습니다.",
            "payload": {
                "changed_count": 0,
                "kept_count": 0,
                "changes": [],
                "results": [],
            },
        }
        return parsed_usage, classifier_details

    indexed_items: list[tuple[int, dict[str, Any]]] = [
        (index, item) for index, item in enumerate(line_items, start=1)
    ]
    classifier_rows = [
        {
            "row_id": row_id,
            "given_category_code": item.get("category_code"),
            "item_name": item.get("item_name"),
        }
        for row_id, item in indexed_items
    ]
    review_response = review_usage_statement(
        usage_statement_id=usage_file_id,
        rows=classifier_rows,
        basic_info={},
    )
    review_map = {result.row_id: result for result in review_response.results}

    changed_count = 0
    kept_count = 0
    updated_items: list[dict[str, Any]] = []
    classifier_results: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for row_id, item in indexed_items:
        original_category = item.get("category_code")
        review = review_map.get(row_id)
        updated_item = dict(item)

        if review is None:
            kept_count += 1
            classifier_results.append(
                {
                    "row_id": row_id,
                    "item_name": item.get("item_name"),
                    "original_category_code": original_category,
                    "final_category_code": original_category,
                    "status": "appropriate",
                    "reason": "classifier result was missing, so the OCR category was kept.",
                }
            )
            updated_items.append(updated_item)
            continue

        updated_category = review.final_category_code or original_category
        status = "appropriate" if review.decision_status == "유지" else "inappropriate"
        if updated_category != original_category:
            changed_count += 1
            changes.append(
                {
                    "row_id": row_id,
                    "item_name": review.item_name,
                    "before": {"category_code": original_category},
                    "after": {"category_code": updated_category},
                    "reason": review.reason,
                }
            )
        else:
            kept_count += 1

        updated_item["category_code"] = updated_category

        classifier_results.append(
            {
                "row_id": row_id,
                "item_name": review.item_name,
                "original_category_code": original_category,
                "final_category_code": updated_category,
                "status": status,
                "reason": review.reason,
            }
        )
        updated_items.append(updated_item)

    classified_usage = dict(parsed_usage)
    classified_usage["line_items"] = updated_items

    summary = (
        f"세부내역 {changed_count}건을 올바른 항목으로 이동했습니다."
        if changed_count
        else "세부내역 분류 이동 없음"
    )
    classifier_details = {
        "event": "classification_updated" if changed_count else "classification_checked",
        "summary": summary,
        "payload": {
            "changed_count": changed_count,
            "kept_count": kept_count,
            "changes": changes,
            "results": classifier_results,
        },
    }
    return classified_usage, classifier_details


# ─────────────────────────────────────────────────────────────
# 엔드포인트 1: 사용내역서 파싱 (/ocr/parse)
# ─────────────────────────────────────────────────────────────

def parse_usage_statement(usage_file_id: int) -> dict[str, Any]:
    """
    사용내역서 PDF를 파싱하고 DB에 저장한다.
    Spring이 파일 업로드 직후 즉시 호출한다.

    Args:
        usage_file_id : 사용내역서 파일의 files.id (DB PK)

    Returns:
        {
          "usage_statement_id": int,
          "parse_status": str,       # SUCCESS / PARTIAL / FAILED
          "item_count": int,
          "elapsed_sec": float,
        }

    Raises:
        ValueError : 파일을 찾을 수 없는 경우
        RuntimeError: 파싱 실패 또는 S3 접근 오류
    """
    start = time.time()
    tmp_paths: list[str] = []
    classifier_log_id: int | None = None
    usage_statement_id: int | None = None
    parsed_usage: dict[str, Any] | None = None
    classifier_details: dict[str, Any] | None = None
    line_items: list[dict[str, Any]] = []

    try:
        with get_connection() as conn:

            # ── DB에서 파일 정보 조회 ──────────────────────────────────────
            file_map = get_files_by_ids(conn, [usage_file_id])
            usage_file = file_map.get(usage_file_id)
            if not usage_file:
                raise ValueError(f"사용내역서 파일을 찾을 수 없습니다 (file_id={usage_file_id})")
            project_id = usage_file["project_id"]

            # ── S3 fetch + 파싱 ────────────────────────────────────────────
            usage_suffix = Path(usage_file["original_filename"]).suffix.lower()
            usage_tmp = _fetch_and_save_temp(usage_file["storage_key"], usage_suffix)
            tmp_paths.append(usage_tmp)

            try:
                parsed_usage = parse_usage_pdf(usage_tmp)
            except Exception as e:
                raise RuntimeError(f"사용내역서 파싱 실패: {e}") from e

            if parsed_usage.get("parse_status") == "FAILED":
                raise RuntimeError("사용내역서 파싱 실패 (FAILED)")

            with get_connection() as classifier_log_conn:
                classifier_log_id = insert_agent_log(
                    classifier_log_conn,
                    project_id=project_id,
                    usage_statement_id=None,
                    details={"source_file_id": usage_file_id},
                    agent_type_code="classi",
                    model_name="classifier_agent",
                )

            parsed_usage, classifier_details = _classify_usage_statement(
                usage_file_id=usage_file_id,
                parsed_usage=parsed_usage,
            )

            # ── DB 저장 ────────────────────────────────────────────────────
            usage_statement_id = insert_usage_statement(
                conn, project_id, usage_file_id, parsed_usage
            )

            insert_usage_statement_summaries(
                conn,
                usage_statement_id,
                parsed_usage.get("category_summaries") or [],
            )

            line_items = parsed_usage.get("line_items") or parsed_usage.get("items") or []
            insert_usage_statement_items(conn, usage_statement_id, line_items)

            # ── 로그 completed ─────────────────────────────────────────────
            if classifier_log_id is not None:
                update_agent_log_status(
                    conn,
                    log_id=classifier_log_id,
                    status_code="completed",
                    details=classifier_details or {},
                )

    except Exception:
        with get_connection() as err_conn:
            if classifier_log_id is not None:
                update_agent_log_status(err_conn, log_id=classifier_log_id, status_code="failed")
        raise

    finally:
        _cleanup(*tmp_paths)

    elapsed = round(time.time() - start, 2)
    classifier_changed_count = sum(
        1
        for result in (((classifier_details or {}).get("payload") or {}).get("results") or [])
        if result.get("original_category_code") != result.get("final_category_code")
    )

    return {
        "usage_statement_id": usage_statement_id,
        "parse_status":       (parsed_usage or {}).get("parse_status", "SUCCESS"),
        "item_count":         len(line_items),
        "classifier_changed_count": classifier_changed_count,
        "elapsed_sec":        elapsed,
    }


# ─────────────────────────────────────────────────────────────
# 엔드포인트 2: 영수증 OCR + 매칭 (/link/run)
# ─────────────────────────────────────────────────────────────

def run_link_pipeline(
    usage_statement_id: int,
    receipt_file_ids: list[int],
    tax_invoice_file_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    영수증·세금계산서 OCR 후 사용내역서와 2-way 매칭하고 결과를 DB에 저장한다.
    분류 Agent + safety-doc 완료 후 Spring이 호출한다.

    Args:
        usage_statement_id   : 이미 저장된 usage_statements.id
        receipt_file_ids     : 영수증·거래명세표 파일들의 files.id 목록
        tax_invoice_file_ids : 세금계산서 파일들의 files.id 목록 (선택)

    Returns:
        {
          "usage_statement_id": int,
          "summary": {
            "total": int, "matched": int,
            "review_needed": int, "unmatched": int, "rejected": int
          },
          "match_results": [...],
          "elapsed_sec": float,
        }

    Raises:
        ValueError : 파일 또는 usage_statement를 찾을 수 없는 경우
        RuntimeError: OCR 실패 또는 S3 접근 오류
    """
    tax_invoice_file_ids = tax_invoice_file_ids or []
    start = time.time()
    tmp_paths: list[str] = []

    # ── 로그 시작 (별도 커넥션으로 즉시 커밋) ─────────────────────────
    # usage_statement → project_id 조회
    with get_connection() as log_conn:
        with log_conn.cursor() as _cur:
            _cur.execute(
                "SELECT project_id FROM usage_statements WHERE id = %s",
                (usage_statement_id,),
            )
            row = _cur.fetchone()
        if row is None:
            raise ValueError(f"usage_statement를 찾을 수 없습니다 (id={usage_statement_id})")
        project_id: int = row[0]
        log_id: int = insert_agent_log(
            log_conn,
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            details={
                "usage_statement_id": usage_statement_id,
                "receipt_file_ids":   receipt_file_ids,
            },
        )

    try:
        with get_connection() as conn:

            # ── DB에서 파일 정보 일괄 조회 ────────────────────────────────
            all_ids = receipt_file_ids + tax_invoice_file_ids
            file_map = get_files_by_ids(conn, all_ids)

            # ── usage_statement items 조회 (매칭용) ───────────────────────
            # v_usage_statement_context 뷰 사용 (직접 테이블 조회 대신)
            with conn.cursor() as _cur:
                _cur.execute(
                    """
                    SELECT item_id, category_code, used_on, item_name, total_amount
                    FROM v_usage_statement_context
                    WHERE usage_statement_id = %s
                      AND item_id IS NOT NULL
                    """,
                    (usage_statement_id,),
                )
                items_rows = _cur.fetchall()
            line_items_for_match = [
                {
                    "line_id":  str(r[0]),
                    "항목코드": _from_category_code(r[1]),
                    "사용일자": str(r[2]) if r[2] else None,
                    "사용내역": r[3],
                    "금액":     r[4],
                }
                for r in items_rows
            ]
            parsed_usage_stub = {"line_items": line_items_for_match}

            # ── 영수증/거래명세표 OCR ──────────────────────────────────────
            receipt_ocr_results: list[dict] = []
            receipt_file_id_map: dict[str, int] = {}  # receipt_id → file_id

            for fid in receipt_file_ids:
                file_info = file_map.get(fid)
                if not file_info:
                    continue

                suffix = Path(file_info["original_filename"]).suffix.lower()
                if suffix not in SUPPORTED_EXTS:
                    continue

                tmp_path = _fetch_and_save_temp(file_info["storage_key"], suffix)
                tmp_paths.append(tmp_path)

                raw        = call_clova_receipt(tmp_path, CLOVA_OCR_SECRET, CLOVA_OCR_URL)
                ocr_result = parse_clova_response(raw)
                ocr_result = validate_ocr_result(ocr_result)
                ocr_result["source_file"] = file_info["original_filename"]

                receipt_file_id_map[ocr_result.get("receipt_id", "")] = fid
                receipt_ocr_results.append(ocr_result)

                time.sleep(0.3)  # CLOVA API rate limit 방지

            # ── 세금계산서 파싱 ────────────────────────────────────────────
            for fid in tax_invoice_file_ids:
                file_info = file_map.get(fid)
                if not file_info:
                    continue

                suffix = Path(file_info["original_filename"]).suffix.lower()
                if suffix not in TAX_INVOICE_EXTS:
                    continue

                tmp_path = _fetch_and_save_temp(file_info["storage_key"], suffix)
                tmp_paths.append(tmp_path)

                parse_tax_invoice(tmp_path, secret=CLOVA_OCR_SECRET, url=CLOVA_OCR_URL)
                time.sleep(0.2)

            # ── 2-way 매칭 ────────────────────────────────────────────────
            batch = match_all_usage_to_receipts(
                usage_statement=parsed_usage_stub,
                receipts=receipt_ocr_results,
                threshold=THRESHOLD_REVIEW,
                threshold_matched=THRESHOLD_MATCHED,
            )

            # ── 매칭 결과 DB 저장 ──────────────────────────────────────────
            for match_result in batch.get("results") or []:
                line_id    = match_result.get("line_id")
                status     = match_result.get("match_status")
                receipt_id = match_result.get("matched_receipt_id")

                if status == "matched" and receipt_id and receipt_id in receipt_file_id_map:
                    file_id_for_receipt = receipt_file_id_map[receipt_id]
                    # line_id는 이미 DB id(str)
                    item_db_id = int(line_id) if line_id and str(line_id).isdigit() else None

                    if item_db_id:
                        receipt_file_info = file_map.get(file_id_for_receipt, {})
                        evidence_type = receipt_file_info.get("uploaded_evidence_type_code", "receipt")

                        insert_evidence_file_link(
                            conn,
                            usage_statement_item_id=item_db_id,
                            file_id=file_id_for_receipt,
                            evidence_type_code=evidence_type,
                        )

                    update_file_status(conn, file_id_for_receipt, "matched")

                elif status in ("unmatched", "rejected") and receipt_id and receipt_id in receipt_file_id_map:
                    file_id_for_receipt = receipt_file_id_map[receipt_id]
                    update_file_status(conn, file_id_for_receipt, "unmatched")

            # ── 로그 completed ─────────────────────────────────────────────
            summary = batch.get("summary") or {}
            update_agent_log_status(
                conn,
                log_id=log_id,
                status_code="completed",
                details={
                    "match_summary":      summary,
                    "usage_statement_id": usage_statement_id,
                    "receipt_file_ids":   receipt_file_ids,
                    "match_results": [
                        {
                            "line_id":      r.get("line_id"),
                            "match_status": r.get("match_status"),
                            "score":        r.get("score"),
                            "gate_failed":  r.get("gate_failed"),
                        }
                        for r in (batch.get("results") or [])
                    ],
                },
            )

    except Exception:
        with get_connection() as err_conn:
            update_agent_log_status(err_conn, log_id=log_id, status_code="failed")
        raise

    finally:
        _cleanup(*tmp_paths)

    elapsed = round(time.time() - start, 2)

    return {
        "usage_statement_id": usage_statement_id,
        "summary":            summary,
        "match_results":      batch.get("results") or [],
        "elapsed_sec":        elapsed,
    }
