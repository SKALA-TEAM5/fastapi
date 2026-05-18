from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.repositories.db import get_connection
from src.services import usage_statement_pipeline_service as pipeline_service
from src.ocr.parse_usage_statement import parse_pdf as parse_usage_pdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="로컬 사용내역서 PDF로 usage statement pipeline 스모크 테스트를 실행한다."
    )
    parser.add_argument(
        "--pdf",
        default="examples/ocr/official_사용내역서_2025년3월.pdf",
        help="테스트할 로컬 PDF 경로",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="files row에 사용할 project_id. 미입력 시 첫 프로젝트를 사용한다.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="files row에 사용할 uploaded_by_user_id. 미입력 시 첫 사용자를 사용한다.",
    )
    return parser.parse_args()


def _resolve_ids(project_id: int | None, user_id: int | None) -> tuple[int, int]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if project_id is None:
                cur.execute("SELECT id FROM projects ORDER BY id LIMIT 1")
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("projects 테이블에 데이터가 없습니다.")
                project_id = int(row[0])

            if user_id is None:
                cur.execute("SELECT id FROM users ORDER BY id LIMIT 1")
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("users 테이블에 데이터가 없습니다.")
                user_id = int(row[0])

            cur.execute(
                "SELECT 1 FROM evidence_types WHERE code = %s",
                ("usage_statement",),
            )
            if cur.fetchone() is None:
                raise RuntimeError("evidence_types에 code='usage_statement'가 없습니다.")

    return project_id, user_id


def _insert_file_row(pdf_path: Path, project_id: int, user_id: int) -> int:
    storage_key = f"local-smoke/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{pdf_path.name}"
    size_bytes = pdf_path.stat().st_size

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files (
                    project_id,
                    uploaded_by_user_id,
                    uploaded_evidence_type_code,
                    original_filename,
                    storage_key,
                    mime_type,
                    size_bytes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    project_id,
                    user_id,
                    "usage_statement",
                    pdf_path.name,
                    storage_key,
                    "application/pdf",
                    size_bytes,
                ),
            )
            row = cur.fetchone()

    if row is None:
        raise RuntimeError("files row INSERT에 실패했습니다.")
    return int(row[0])


def _load_usage_statement_snapshot(usage_statement_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_id, source_file_id, report_month, revision_no,
                       document_written_date, cumulative_progress_rate, status_code
                FROM usage_statements
                WHERE id = %s
                """,
                (usage_statement_id,),
            )
            usage_statement_row = cur.fetchone()

            cur.execute(
                """
                SELECT id, category_code, used_on, item_name, quantity, unit_price, total_amount
                FROM usage_statement_items
                WHERE usage_statement_id = %s
                ORDER BY id
                """,
                (usage_statement_id,),
            )
            item_rows = cur.fetchall()

            cur.execute(
                """
                SELECT id, agent_type_code, status_code, details, created_at
                FROM agent_logs
                WHERE agent_type_code = 'classi'
                ORDER BY id DESC
                LIMIT 1
                """
            )
            classi_log = cur.fetchone()

    return {
        "usage_statement": usage_statement_row,
        "items": item_rows,
        "classi_log": classi_log,
    }


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    project_id, user_id = _resolve_ids(args.project_id, args.user_id)
    file_id = _insert_file_row(pdf_path, project_id, user_id)

    parsed_preview = parse_usage_pdf(str(pdf_path))
    line_items = parsed_preview.get("line_items") or []
    missing_name_rows = [
        {
            "line_no": item.get("line_no"),
            "page_no": item.get("page_no"),
            "category_code": item.get("category_code"),
            "used_on": item.get("used_on"),
            "item_name": item.get("item_name"),
            "total_amount": item.get("total_amount"),
            "remark": item.get("remark"),
        }
        for item in line_items
        if not item.get("item_name")
    ]
    print(json.dumps(
        {
            "parsed_preview": {
                "parse_status": parsed_preview.get("parse_status"),
                "line_item_count": len(line_items),
                "missing_item_name_count": len(missing_name_rows),
                "missing_item_name_rows": missing_name_rows,
            }
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ))

    original_fetch_file = pipeline_service.fetch_file
    pipeline_service.fetch_file = lambda _storage_key: pdf_path.read_bytes()
    try:
        result = pipeline_service.parse_usage_statement(file_id)
    finally:
        pipeline_service.fetch_file = original_fetch_file

    snapshot = _load_usage_statement_snapshot(int(result["usage_statement_id"]))
    print(json.dumps(
        {
            "file_id": file_id,
            "pipeline_result": result,
            "db_snapshot": snapshot,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
