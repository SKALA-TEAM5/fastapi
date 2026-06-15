# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 엔드포인트 ]
#
# /validator/usage-statement
#   - 설명 : 사용내역서 전체 법령 검증 + agent_logs 항목별 INSERT
#   - Body : {
#               "사용내역서ID": int,
#               "기본정보": { "산안비총액": float, "누계공정률": float },
#               "카테고리별데이터": [ ... ]
#             }
#   - 호출 : validate_usage_statement_service()
#   - 적재 : agent_logs INSERT (legal, 항목당 1행)
#
# /validator/document
#   - 설명 : 단일 카테고리 법령 검증 (DB 적재 없음, 결과만 반환)
#   - Body : { "category": str, "items": dict, "base_amount": float, ... }
#   - 호출 : validate_document_service()
#
# /validator/audit
#   - 설명 : 복수 카테고리 묶음 법령 검증 (DB 적재 없음, 결과만 반환)
#   - Body : { "base_amount": float, "categories": { ... } }
#   - 호출 : run_audit_service()
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. run_audit_service()               : 카테고리 묶음 검토 서비스
# 2. validate_document_service()       : 단일 카테고리 검토 서비스
# 3. validate_usage_statement_service(): 사용내역서 검토 서비스
# 4. _write_legal_agent_log()          : agent_logs INSERT (legal 전용)
# 5. _derive_log_result()              : 검증 결과에서 result_code/reason 파생
# 6. _build_log_details()              : agent_logs.details JSONB 구성
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from threading import Lock
from typing import Any

try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:  # pragma: no cover
    get_openai_callback = None  # type: ignore

from src.agents.validator_agent.audit import decide_category
from src.agents.validator_agent.calculator import calculate_category_metrics
from src.agents.validator_agent.context_retriever import retrieve_category_context
from src.agents.validator_agent.parser import (
    build_legacy_blocks,
    parse_single_category_request,
    parse_usage_statement,
)
from src.agents.validator_agent.rule_matcher import match_category_rules
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.classifier import CATEGORIES as _CATEGORIES
from src.schemas.validator import AuditResponse, CategoryAuditResult

logger = logging.getLogger(__name__)


def run_audit_service(
    *,
    base_amount: float,
    categories: dict[str, dict[str, float]],
    collection: str = DEFAULT_COLLECTION,
    basic_info_by_category: dict[str, dict[str, Any]] | None = None,
    summaries_by_category: dict[str, dict[str, Any]] | None = None,
    progress_rate: float | None = None,
) -> AuditResponse:
    parsed = build_legacy_blocks(
        base_amount=base_amount,
        categories=categories,
        basic_info_by_category=basic_info_by_category,
        summaries_by_category=summaries_by_category,
        progress_rate=progress_rate,
    )
    return _validate_blocks(parsed.base_amount, parsed.blocks, collection=collection)


def validate_document_service(
    *,
    category: str,
    items: dict[str, float] | None = None,
    basic_info: dict[str, Any] | None = None,
    base_amount: float | None = None,
    document: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> CategoryAuditResult:
    block = parse_single_category_request(
        category=category,
        items=items,
        basic_info=basic_info,
        base_amount=base_amount,
        document=document,
    )
    response = _validate_blocks(block.base_amount, [block], collection=collection)
    return response.categories[block.category_name]


def validate_usage_statement_service(
    *,
    document: dict[str, Any],
    collection: str = DEFAULT_COLLECTION,
    project_id: int | None = None,
    usage_statement_id: int | None = None,
    model_name: str = "claude-sonnet-4-6",
) -> AuditResponse:
    parsed = parse_usage_statement(document)

    # 토큰 집계는 _validate_blocks 내부에서 per-thread get_openai_callback()으로 처리한다.
    # match_category_rules도 동일하게 per-thread 방식으로 집계하므로
    # 중첩 ThreadPoolExecutor 구조에서 토큰 누락이 발생하지 않는다.
    token_bucket: dict[str, int] = {"total": 0}
    response = _validate_blocks(
        parsed.base_amount, parsed.blocks, collection=collection, token_bucket=token_bucket
    )
    total_tokens: int | None = token_bucket["total"] or None

    if project_id is not None and usage_statement_id is not None:
        _write_legal_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            response=response,
            model_name=model_name,
            token=total_tokens,
        )
    return response


# ── agent_logs 기록 ───────────────────────────────────────────────────────────
#
# 행 구성:
#   항목 행   : usage_statement_item_id = 항목 DB ID  (항목당 1행)
#
# result_code 우선순위: hil(항목불허/검토필요/카테고리 위반) > success
# --------------------------------------------------------------------------

_INSERT_AGENT_LOG_SQL = """
    INSERT INTO agent_logs (
        project_id, usage_statement_id, usage_statement_item_id,
        agent_type_code, status_code, result_code,
        reason, details, model_name, token
    )
    VALUES (
        %(project_id)s, %(usage_statement_id)s, %(item_db_id)s,
        'legal', 'success', %(result_code)s,
        %(reason)s, %(details)s::jsonb, %(model_name)s, %(token)s
    )
    RETURNING id
"""


def _write_legal_agent_log(
    *,
    project_id: int,
    usage_statement_id: int,
    response: AuditResponse,
    model_name: str = "claude-sonnet-4-6",
    token: int | None = None,
) -> None:
    """
    legal 에이전트 검증 결과를 agent_logs에 INSERT한다.

    - 항목 행: 카테고리별 항목당 1행 (usage_statement_item_id = 항목 DB ID)

    usage_statement_items에서 (category_code, item_name)으로 DB ID를 조회한다.
    매칭 실패 시 usage_statement_item_id = NULL로 기록.
    """
    from src.repositories.db import get_connection

    try:
        with get_connection() as conn:
            # ── 항목 ID 맵 선조회 (1 query) ──────────────────────────────
            item_id_map = _fetch_item_id_map(conn, usage_statement_id)

            inserted = 0
            with conn.cursor() as cur:
                for _, result in response.categories.items():
                    for item in result.items:
                        item_db_id = _resolve_item_db_id(item_id_map, item)
                        result_code = _item_result_code(item, result)
                        reason = _item_reason(item, result)
                        details = _item_details(item, result)

                        cur.execute(
                            _INSERT_AGENT_LOG_SQL,
                            {
                                "project_id": project_id,
                                "usage_statement_id": usage_statement_id,
                                "item_db_id": item_db_id,
                                "result_code": result_code,
                                "reason": reason,
                                "details": json.dumps(details, ensure_ascii=False),
                                "model_name": model_name,
                                "token": token,
                            },
                        )
                        inserted += 1

        logger.info(
            "[agent_log] legal INSERT %d건 완료 (카테고리 summary 포함)", inserted
        )
    except Exception as exc:
        logger.warning("[agent_log] INSERT 실패 (로그 생략 후 계속): %s", exc)


def _fetch_item_id_map(conn, usage_statement_id: int) -> dict[tuple[str, str], int]:
    """
    usage_statement_items에서 (category_code, item_name) → id 맵을 반환한다.
    item_name이 중복인 경우 가장 작은 id 우선.
    """
    sql = """
        SELECT id, category_code, item_name
        FROM usage_statement_items
        WHERE usage_statement_id = %s
        ORDER BY id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (usage_statement_id,))
        result: dict[tuple[str, str], int] = {}
        for row_id, cat_code, item_name in cur.fetchall():
            key = (cat_code, item_name)
            if key not in result:  # 중복 시 첫 번째(작은 id) 우선
                result[key] = row_id
    return result


def _resolve_item_db_id(item_id_map: dict[tuple[str, str], int], item) -> int | None:
    """Match validator item category names back to usage_statement_items.category_code."""
    item_name = getattr(item, "item", "")
    category = getattr(item, "category", "")
    category_code = _category_code_for(category)

    for key in ((category_code, item_name), (category, item_name)):
        if key[0] and key in item_id_map:
            return item_id_map[key]
    return None


def _category_code_for(category: str) -> str:
    if category in _CATEGORIES:
        return category
    return next(
        (code for code, name in _CATEGORIES.items() if name == category), category
    )


def _item_result_code(item, result: "CategoryAuditResult") -> str:
    """
    항목 행 result_code 결정.

    우선순위:
      1. 항목 자체 불허                  → hil
      2. 항목 자체 검토 필요              → hil
      3. 카테고리 한도 초과/공정률 부족    → hil
      4. 정상                            → success

    summary row를 별도로 만들지 않으므로 카테고리 차원의 수치 위반도
    해당 카테고리 item row에 반영한다.
    """
    # 항목 자체가 불허인 경우만 hil — allowed: true면 공정률/한도/검토필요는 무시
    if item.allowed is False:
        return "hil"
    if getattr(result, "exceeded", False):
        return "hil"
    return "success"


def _category_result_code(result) -> str:
    """카테고리 레벨 result_code: 부적절이면 hil, 그 외 success."""
    status = getattr(result, "status", "") or ""
    if status == "부적절":
        return "hil"
    shortfall = getattr(result, "usage_shortfall_amount", None)
    if shortfall is not None and shortfall > 0:
        return "hil"
    if getattr(result, "exceeded", False):
        return "hil"
    return "success"


def _category_reason(result) -> str:
    """카테고리 레벨 reason: rejection_reason 우선, 없으면 공정률/한도 정보."""
    reason = getattr(result, "rejection_reason", "") or ""
    if reason:
        return reason[:1000]
    shortfall = getattr(result, "usage_shortfall_amount", None)
    if shortfall and shortfall > 0:
        rate = getattr(result, "required_usage_rate", None)
        return f"공정률 기준 미달: 누적사용액이 기준({rate * 100:.0f}%)에 {shortfall:,.0f}원 부족합니다."[
            :1000
        ]
    if getattr(result, "exceeded", False):
        limit = getattr(result, "limit", None)
        return (
            f"한도 초과: 카테고리 사용한도({limit:,.0f}원)를 초과하였습니다."[:1000]
            if limit
            else "한도 초과"
        )
    return ""


def _category_details(category_name: str, result) -> dict:
    """카테고리 레벨 agent_logs.details."""
    return {
        "category": {
            "category_name": category_name,
            "status": getattr(result, "status", None),
            "total": getattr(result, "total", None),
            "limit": getattr(result, "limit", None),
            "exceeded": getattr(result, "exceeded", False),
            "progress_rate": getattr(result, "progress_rate", None),
            "required_usage_rate": getattr(result, "required_usage_rate", None),
            "required_used_amount": getattr(result, "required_used_amount", None),
            "cumulative_used_amount": getattr(result, "cumulative_used_amount", None),
            "usage_shortfall_amount": getattr(result, "usage_shortfall_amount", None),
        }
    }


_DUPLICATE_CONCLUSION_RE = re.compile(
    r"(허용됩니다|사용이 가능합니다|불허됩니다|사용이 불가합니다)\s*[.\s]+"
    r"(허용됩니다|사용이 가능합니다|불허됩니다|사용이 불가합니다)[.\s]*$"
)
# 공백 포함 조항 번호 패턴 — "제X조 제X항 제X호" 전체 매칭
_BARE_ARTICLE_RE = re.compile(r"(?<![가-힣])(제\d+조(?:\s*제\d+항)?(?:\s*제\d+호)?)")


def _enrich_reason_law(reason: str, referenced_laws: list[str]) -> str:
    """reason 내 bare 조항 번호(법령명 없음)를 referenced_laws의 완전 표기로 교체.
    이미 법령명이 붙은 조항은 건드리지 않는다.
    """
    if not referenced_laws:
        return reason

    full_law_refs = [l for l in referenced_laws if re.search(r"[가-힣]{2,}", l)]
    if not full_law_refs:
        return reason

    # reason을 한 번만 스캔 — 이미 법령명이 있는 구간은 보호
    def _replace(m: re.Match) -> str:
        bare = m.group(1).replace(" ", "")
        start = m.start()
        # 앞 30자에 한국어 2자 이상 있으면 이미 법령명이 있는 것 → 스킵
        preceding = reason[max(0, start - 30) : start]
        if re.search(r"[가-힣]{2,}", preceding):
            return m.group(0)
        # 가장 길게 매칭되는 full law ref 찾기 (세부 조항 우선)
        best = None
        for law in sorted(full_law_refs, key=len, reverse=True):
            law_norm = law.replace(" ", "")
            if bare in law_norm and law_norm != bare:
                best = law
                break
            # bare가 law의 기본 조항(제X조)과 같을 때도 매핑
            if bare.startswith("제") and law_norm.startswith(bare):
                best = law
                break
        return best if best else m.group(0)

    return _BARE_ARTICLE_RE.sub(_replace, reason)


def _category_metrics_suffix(result) -> str:
    """카테고리 사용한도 정보를 reason에 추가할 접미 문장 생성.
    공정률 shortfall은 사용내역서 전체 기준이므로 개별 항목 reason에 포함하지 않는다.
    """
    limit = getattr(result, "limit", None)
    exceeded = getattr(result, "exceeded", False)
    if limit is None or not exceeded:
        return ""
    return f"이 카테고리 사용한도는 {limit:,.0f}원이며, 현재 한도를 초과하였습니다."


def _item_reason(item, result: "CategoryAuditResult" | None = None) -> str:
    """rule_matcher에서 생성된 reason_text를 agent_logs.reason에 저장한다."""
    reason = getattr(item, "reason_text", "") or getattr(item, "reasoning", "") or ""
    # 결론 중복 제거
    reason = _DUPLICATE_CONCLUSION_RE.sub(r"\1.", reason).strip()
    # bare 조항 번호 → 법령명 포함으로 보완
    referenced_laws: list[str] = list(getattr(item, "referenced_laws", []) or [])
    if referenced_laws:
        reason = _enrich_reason_law(reason, referenced_laws)
    return reason[:1000]


def _item_details(item, result: "CategoryAuditResult") -> dict:
    """
    항목 행 agent_logs.details JSONB.

    DESIGN_DECISIONS.md P6 기준 Legal agent 형식:
    {
        "categories": {
            "CAT_XX": {
                "status":           "ok" | "supplement",
                "supplement_codes": ["LEGAL_VIOLATION"] | ["LEGAL_INSUFFICIENT"] | [],
                "basis":            "제7조제1항제2호",   ← 첫 번째 참조 조항
                "detail":           None,
                "confidence":       0.85
            }
        },
        "item": {
            "item_name":        "사무실 소화기",
            "amount":           500000,
            "allowed":          false,
            "referenced_laws":  [...],
            "judgment_source":  "law_rule" | "llm_fallback" | ...
        }
    }
    """
    # ── supplement_codes 결정 ─────────────────────────────────────────────────
    is_violation = item.allowed is False
    # 공정률 shortfall은 문서 레벨 판단이므로 개별 항목 supplement_codes에 반영 안 함
    # 한도 초과(exceeded)만 항목 레벨 카테고리 이슈로 인정
    has_category_issue = bool(getattr(result, "exceeded", False))
    is_insufficient = (
        bool(getattr(item, "needs_human_review", False)) or has_category_issue
    )

    if is_violation:
        status = "supplement"
        supplement_codes: list[str] = ["LEGAL_VIOLATION"]
    elif is_insufficient:
        status = "supplement"
        supplement_codes = ["LEGAL_INSUFFICIENT"]
    else:
        status = "ok"
        supplement_codes = []

    # ── basis: 항목 참조 조항 첫 번째 ────────────────────────────────────────
    item_laws: list[str] = list(getattr(item, "referenced_laws", []) or [])
    basis = item_laws[0] if item_laws else ""

    # ── confidence: 항목 confidence ───────────────────────────────────────────
    confidence = round(float(getattr(item, "confidence", 0.8)), 2)

    # ── category_code 조회 ───────────────────────────────────────────────────
    cat_code = _category_code_for(getattr(item, "category", ""))

    cat_entry: dict = {
        "status": status,
        "supplement_codes": supplement_codes,
        "detail": None,
        "confidence": confidence,
    }
    if basis:
        cat_entry["basis"] = basis

    # ── item section ──────────────────────────────────────────────────────────
    item_section: dict = {
        "item_name": item.item,
        "amount": int(item.amount),
        "allowed": item.allowed,
    }
    if item_laws:
        item_section["referenced_laws"] = item_laws
    judgment_source = getattr(item, "judgment_source", None)
    if judgment_source:
        item_section["judgment_source"] = judgment_source

    # ── category metrics (한도, 공정률) ─────────────────────────────────────────
    cat_metrics: dict = {}
    limit = getattr(result, "limit", None)
    exceeded = getattr(result, "exceeded", False)
    progress_rate = getattr(result, "progress_rate", None)
    shortfall = getattr(result, "usage_shortfall_amount", None)
    cumulative = getattr(result, "cumulative_used_amount", None)
    required = getattr(result, "required_used_amount", None)
    if limit is not None:
        cat_metrics["limit"] = int(limit)
    if exceeded:
        cat_metrics["exceeded"] = True
    if progress_rate is not None:
        cat_metrics["progress_rate"] = progress_rate
    if shortfall is not None and shortfall > 0:
        cat_metrics["usage_shortfall_amount"] = int(shortfall)
    if cumulative is not None:
        cat_metrics["cumulative_used_amount"] = int(cumulative)
    if required is not None:
        cat_metrics["required_used_amount"] = int(required)
    if cat_metrics:
        cat_entry["metrics"] = cat_metrics

    return {
        "categories": {cat_code: cat_entry} if cat_code else {},
        "item": item_section,
    }


# ── 카테고리 검증 실행 ────────────────────────────────────────────────────────


def _validate_blocks(
    base_amount: float,
    blocks,
    *,
    collection: str,
    token_bucket: dict[str, int] | None = None,
) -> AuditResponse:
    # 공정률 shortfall은 전체 카테고리 누적합 기준으로 판단한다.
    total_cumulative_used_amount: float | None = None
    for block in blocks:
        val = block.summary.get("누적사용금액") or block.summary.get(
            "cumulative_amount"
        )
        try:
            v = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        total_cumulative_used_amount = (total_cumulative_used_amount or 0.0) + v

    # 카테고리별 토큰은 match_category_rules의 CategoryRuleBundle.token_usage에 집계된다.
    # 카테고리 스레드는 copy_context() 없이 독립 실행하고, 토큰 합계는 token_bucket으로 전달.
    results: dict[str, CategoryAuditResult] = {}
    token_lock = Lock()

    def _run_category(block) -> CategoryAuditResult:
        result = _validate_category_block(block, collection, total_cumulative_used_amount)
        if token_bucket is not None:
            with token_lock:
                token_bucket["total"] += result.token_usage
        return result

    with ThreadPoolExecutor(max_workers=min(max(len(blocks), 1), 2)) as executor:
        futures = {
            executor.submit(_run_category, block): block.category_name
            for block in blocks
        }
        for future in as_completed(futures):
            category_name = futures[future]
            results[category_name] = future.result()
    total_token = token_bucket["total"] if token_bucket is not None else 0
    return AuditResponse(base_amount=base_amount, categories=results, total_token_usage=total_token)


def _validate_category_block(
    block, collection: str, total_cumulative_used_amount: float | None = None
) -> CategoryAuditResult:
    retrieved = retrieve_category_context(block=block, collection=collection)
    rule_bundle = match_category_rules(block=block, retrieved=retrieved)
    computation = calculate_category_metrics(
        block=block,
        rule_bundle=rule_bundle,
        total_cumulative_used_amount=total_cumulative_used_amount,
    )
    result = decide_category(
        block=block,
        retrieved=retrieved,
        rule_bundle=rule_bundle,
        computation=computation,
    )
    for item in result.items:
        item.category = block.category_name
        item.category_limit_pct = rule_bundle.limit_pct
        item.category_limit_rule = rule_bundle.limit_rule
    return result
