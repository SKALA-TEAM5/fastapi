"""
OCR 파이프라인 DB 레포지토리
━━━━━━━━━━━━━━━━━━━━━━━━━━
파이프라인 실행에 필요한 DB 읽기/쓰기 함수 모음.

읽기:
  - get_file_by_id     : file_id → storage_key, evidence_type 등
  - get_files_by_ids   : 여러 file_id 일괄 조회

쓰기 (파싱 결과):
  - insert_usage_statement          : usage_statements INSERT
  - insert_usage_statement_summaries: usage_statement_summaries INSERT
  - insert_usage_statement_items    : usage_statement_items INSERT → {uuid: db_id} 매핑 반환

쓰기 (매칭 결과):
  - update_file_status       : files.status_code UPDATE
  - insert_evidence_file_link: evidence_file_links INSERT
  - insert_validation_log    : validation_logs INSERT
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from psycopg2.extensions import connection as PgConnection
import psycopg2.extras

# ─────────────────────────────────────────────────────────────
# 카테고리 코드 매핑 (OCR JSON 항목코드 → DB category_code)
# OCR 파서: "1"~"9"  /  DB: "CAT_01"~"CAT_09"
# ─────────────────────────────────────────────────────────────
_CATEGORY_CODE_MAP: dict[str, str] = {
    "1": "CAT_01", "2": "CAT_02", "3": "CAT_03",
    "4": "CAT_04", "5": "CAT_05", "6": "CAT_06",
    "7": "CAT_07", "8": "CAT_08", "9": "CAT_09",
}


def _to_category_code(raw: str | None) -> str | None:
    """OCR 항목코드("1"~"9")를 DB category_code("CAT_01"~"CAT_09")로 변환."""
    if raw is None:
        return None
    return _CATEGORY_CODE_MAP.get(str(raw).strip(), raw)


_CATEGORY_CODE_REVERSE: dict[str, str] = {v: k for k, v in _CATEGORY_CODE_MAP.items()}


def _from_category_code(db_code: str | None) -> str | None:
    """DB category_code("CAT_01"~"CAT_09")를 OCR 항목코드("1"~"9")로 역변환."""
    if db_code is None:
        return None
    return _CATEGORY_CODE_REVERSE.get(str(db_code).strip(), db_code)


def _safe_date(value: str | None) -> date | None:
    """YYYY-MM-DD 문자열을 date 객체로 변환. 실패 시 None 반환."""
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _first_day_of_month(d: date) -> date:
    """주어진 날짜의 해당 월 1일을 반환."""
    return d.replace(day=1)


# ═══════════════════════════════════════════════════════════════
# 읽기
# ═══════════════════════════════════════════════════════════════

def get_file_by_id(conn: PgConnection, file_id: int) -> dict[str, Any]:
    """
    files 테이블에서 단일 파일 정보를 조회한다.

    Returns:
        {
          "id": int,
          "project_id": int,
          "storage_key": str,
          "original_filename": str,
          "uploaded_evidence_type_code": str,
          "mime_type": str,
        }

    Raises:
        ValueError: 파일을 찾을 수 없는 경우
    """
    sql = """
        SELECT id, project_id, storage_key, original_filename,
               uploaded_evidence_type_code, mime_type
        FROM files
        WHERE id = %(file_id)s
          AND deleted_at IS NULL
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"file_id": file_id})
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"파일을 찾을 수 없습니다 (file_id={file_id})")
    return dict(row)


def get_already_linked_file_ids(
    conn: PgConnection,
    file_ids: list[int],
) -> set[int]:
    """
    주어진 file_id 목록 중 evidence_file_links에 이미 연결된 file_id를 반환한다.

    상시 업로드 환경에서 동일 영수증이 다른 usage_statement 항목에
    중복 연결되는 것을 방지하기 위해 /link/run 시작 시 호출한다.

    Returns:
        이미 linked된 file_id 집합 (후보 목록에서 제외 대상)
    """
    if not file_ids:
        return set()

    sql = """
        SELECT DISTINCT file_id
        FROM evidence_file_links
        WHERE file_id = ANY(%(ids)s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"ids": file_ids})
        rows = cur.fetchall()

    return {row[0] for row in rows}


def get_files_by_ids(
    conn: PgConnection, file_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """
    여러 file_id를 일괄 조회한다.

    Returns:
        {file_id: file_info_dict, ...}
    """
    if not file_ids:
        return {}

    sql = """
        SELECT id, project_id, storage_key, original_filename,
               uploaded_evidence_type_code, mime_type
        FROM files
        WHERE id = ANY(%(ids)s)
          AND deleted_at IS NULL
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"ids": file_ids})
        rows = cur.fetchall()

    return {row["id"]: dict(row) for row in rows}


# ═══════════════════════════════════════════════════════════════
# 쓰기 — 파싱 결과
# ═══════════════════════════════════════════════════════════════

def insert_usage_statement(
    conn: PgConnection,
    project_id: int,
    source_file_id: int,
    parsed: dict,
) -> int:
    """
    사용내역서 파싱 결과를 usage_statements에 INSERT한다.

    - report_month      : line_items 첫 항목의 사용일자로부터 해당 월 1일 추출
                          (line_items가 없으면 오늘 날짜 기준)
    - document_written_date: 파싱 결과에 없으므로 오늘 날짜로 대체
    - cumulative_progress_rate: header.공정률 (없으면 0)
    - revision_no       : 기본값 1

    Returns:
        생성된 usage_statements.id
    """
    header = parsed.get("header") or {}
    line_items = parsed.get("line_items") or parsed.get("items") or []

    # report_month 추출
    report_month: date
    if line_items:
        first_date = _safe_date(line_items[0].get("사용일자"))
        report_month = _first_day_of_month(first_date) if first_date else date.today().replace(day=1)
    else:
        report_month = date.today().replace(day=1)

    # 누계공정률
    try:
        progress_rate = float(header.get("공정률") or 0)
    except (ValueError, TypeError):
        progress_rate = 0.0

    sql = """
        INSERT INTO usage_statements
            (project_id, source_file_id, report_month, revision_no,
             document_written_date, cumulative_progress_rate)
        VALUES
            (%(project_id)s, %(source_file_id)s, %(report_month)s, %(revision_no)s,
             %(document_written_date)s, %(cumulative_progress_rate)s)
        ON CONFLICT (source_file_id) DO UPDATE
            SET updated_at = now()
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "project_id":               project_id,
            "source_file_id":           source_file_id,
            "report_month":             report_month,
            "revision_no":              1,
            "document_written_date":    date.today(),
            "cumulative_progress_rate": progress_rate,
        })
        row = cur.fetchone()

    return row[0]


def insert_usage_statement_summaries(
    conn: PgConnection,
    usage_statement_id: int,
    category_summaries: list[dict],
) -> None:
    """
    카테고리별 전회/금회/누계 금액을 usage_statement_summaries에 INSERT한다.
    이미 존재하는 경우 금액을 업데이트한다.
    """
    if not category_summaries:
        return

    sql = """
        INSERT INTO usage_statement_summaries
            (usage_statement_id, category_code,
             previous_amount, current_amount, cumulative_amount)
        VALUES
            (%(usage_statement_id)s, %(category_code)s,
             %(previous_amount)s, %(current_amount)s, %(cumulative_amount)s)
        ON CONFLICT (usage_statement_id, category_code) DO UPDATE
            SET previous_amount   = EXCLUDED.previous_amount,
                current_amount    = EXCLUDED.current_amount,
                cumulative_amount = EXCLUDED.cumulative_amount,
                updated_at        = now()
    """
    params_list = []
    for summary in category_summaries:
        cat_code = _to_category_code(summary.get("항목코드"))
        if cat_code is None:
            continue
        params_list.append({
            "usage_statement_id": usage_statement_id,
            "category_code":      cat_code,
            "previous_amount":    int(summary.get("전회금액") or 0),
            "current_amount":     int(summary.get("금회금액") or 0),
            "cumulative_amount":  int(summary.get("누계금액") or 0),
        })

    if params_list:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list)


def insert_usage_statement_items(
    conn: PgConnection,
    usage_statement_id: int,
    line_items: list[dict],
) -> dict[str, int]:
    """
    사용내역 항목들을 usage_statement_items에 INSERT한다.

    Returns:
        OCR line_id(UUID) → DB id 매핑 딕셔너리
        예: {"uuid-abc": 101, "uuid-def": 102, ...}
        매칭 단계에서 evidence_file_links INSERT 시 사용한다.
    """
    if not line_items:
        return {}

    sql = """
        INSERT INTO usage_statement_items
            (usage_statement_id, category_code, used_on, item_name,
             unit, quantity, unit_price, total_amount, remark, page_no)
        VALUES
            (%(usage_statement_id)s, %(category_code)s, %(used_on)s, %(item_name)s,
             %(unit)s, %(quantity)s, %(unit_price)s, %(total_amount)s, %(remark)s, %(page_no)s)
        RETURNING id
    """
    uuid_to_db_id: dict[str, int] = {}

    with conn.cursor() as cur:
        for item in line_items:
            extra = item.get("추가정보") or {}
            cat_code = _to_category_code(item.get("항목코드"))
            used_on = _safe_date(item.get("사용일자"))

            if used_on is None or cat_code is None:
                continue  # 필수 필드 누락 시 스킵

            cur.execute(sql, {
                "usage_statement_id": usage_statement_id,
                "category_code":      cat_code,
                "used_on":            used_on,
                "item_name":          str(item.get("사용내역") or "")[:300],
                "unit":               str(extra.get("단위") or "")[:50] or None,
                "quantity":           float(extra.get("수량") or 0),
                "unit_price":         float(extra.get("단가") or 0),
                "total_amount":       int(item.get("금액") or 0),
                "remark":             None,
                "page_no":            int(item.get("page_no") or 1),
            })
            row = cur.fetchone()
            line_id = item.get("line_id")
            if line_id and row:
                uuid_to_db_id[line_id] = row[0]

    return uuid_to_db_id


# ═══════════════════════════════════════════════════════════════
# 쓰기 — 매칭 결과
# ═══════════════════════════════════════════════════════════════

def update_file_status(
    conn: PgConnection,
    file_id: int,
    status_code: str,
) -> None:
    """
    files.status_code를 업데이트한다.
    status_code: 'matched' | 'unmatched'
    """
    sql = """
        UPDATE files
        SET status_code = %(status_code)s
        WHERE id = %(file_id)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"file_id": file_id, "status_code": status_code})


def insert_evidence_file_link(
    conn: PgConnection,
    usage_statement_item_id: int,
    file_id: int,
    evidence_type_code: str,
) -> None:
    """
    매칭된 항목-파일 연결을 evidence_file_links에 INSERT한다.
    이미 존재하는 경우 무시한다.

    [V7 변경] category_code 컬럼 제거됨.
    category는 usage_statement_items JOIN으로 항상 정확히 얻을 수 있으므로
    evidence_file_links에서 중복 보관하지 않는다.
    """
    sql = """
        INSERT INTO evidence_file_links
            (usage_statement_item_id, file_id, evidence_type_code)
        VALUES
            (%(item_id)s, %(file_id)s, %(evidence_type_code)s)
        ON CONFLICT (usage_statement_item_id, file_id) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "item_id":            usage_statement_item_id,
            "file_id":            file_id,
            "evidence_type_code": evidence_type_code,
        })


def insert_agent_log(
    conn: PgConnection,
    project_id: int,
    usage_statement_id: int | None = None,
    details: dict | None = None,
) -> int:
    """
    파이프라인 시작 시 agent_logs에 running 상태로 INSERT한다.

    V1 agent_logs 컬럼: project_id, usage_statement_id, agent_type_code,
                        status_code, details, model_name
    (usage_statement_item_id / validation_type_code / severity_code 는 없음)

    Returns:
        생성된 agent_logs.id (완료/실패 시 update_agent_log_status에 전달)
    """
    import json as _json

    sql = """
        INSERT INTO agent_logs
            (project_id, usage_statement_id,
             status_code, agent_type_code,
             details, model_name)
        VALUES
            (%(project_id)s, %(usage_statement_id)s,
             'running', 'link',
             %(details)s::jsonb, 'clova_ocr_v2')
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "project_id":         project_id,
            "usage_statement_id": usage_statement_id,
            "details":            _json.dumps(details or {}, ensure_ascii=False),
        })
        row = cur.fetchone()
    return row[0]


def update_agent_log_status(
    conn: PgConnection,
    log_id: int,
    status_code: str,
    details: dict | None = None,
) -> None:
    """
    파이프라인 완료/실패 시 agent_logs의 status_code를 업데이트한다.

    status_code:
        'completed' — 정상 완료
        'failed'    — 서버/시스템 오류 (비즈니스 실패 아님)
    """
    import json as _json

    if details is not None:
        sql = """
            UPDATE agent_logs
            SET status_code = %(status_code)s,
                details     = %(details)s::jsonb
            WHERE id = %(log_id)s
        """
        params = {
            "log_id":      log_id,
            "status_code": status_code,
            "details":     _json.dumps(details, ensure_ascii=False),
        }
    else:
        sql = """
            UPDATE agent_logs
            SET status_code = %(status_code)s
            WHERE id = %(log_id)s
        """
        params = {"log_id": log_id, "status_code": status_code}

    with conn.cursor() as cur:
        cur.execute(sql, params)
