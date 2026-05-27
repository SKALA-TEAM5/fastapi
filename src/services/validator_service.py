# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents.validator_agent.calculator import calculate_category_metrics
from src.agents.validator_agent.context_retriever import retrieve_category_context
from src.agents.validator_agent.audit import decide_category
from src.agents.validator_agent.parser import (
    build_legacy_blocks,
    parse_single_category_request,
    parse_usage_statement,
)
from src.agents.validator_agent.presenter import (
    _synthesize_reason_with_llm as _llm_synthesize_reason,
    _build_sources as _build_cat_sources,
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
    response = _validate_blocks(parsed.base_amount, parsed.blocks, collection=collection)
    if project_id is not None and usage_statement_id is not None:
        _write_legal_agent_log(
            project_id=project_id,
            usage_statement_id=usage_statement_id,
            response=response,
            model_name=model_name,
        )
    return response


# ── agent_logs 기록 ───────────────────────────────────────────────────────────
#
# 행 구성:
#   항목 행   : usage_statement_item_id = 항목 DB ID  (항목당 1행)
#   전체 행   : usage_statement_item_id = NULL         (사용내역서 전체 요약 1행)
#
# result_code 우선순위: hil(항목불허) > hil(한도초과) > hil(공정률부족) > success
# --------------------------------------------------------------------------

_INSERT_AGENT_LOG_SQL = """
    INSERT INTO agent_logs (
        project_id, usage_statement_id, usage_statement_item_id,
        agent_type_code, status_code, result_code,
        reason, details, model_name
    )
    VALUES (
        %(project_id)s, %(usage_statement_id)s, %(item_db_id)s,
        'legal', 'success', %(result_code)s,
        %(reason)s, %(details)s::jsonb, %(model_name)s
    )
    RETURNING id
"""


def _write_legal_agent_log(
    *,
    project_id: int,
    usage_statement_id: int,
    response: AuditResponse,
    model_name: str = "claude-sonnet-4-6",
) -> None:
    """
    legal 에이전트 검증 결과를 agent_logs에 INSERT한다.

    - 항목 행: 카테고리별 항목당 1행 (usage_statement_item_id = 항목 DB ID)
    - 전체 행: 사용내역서 전체 요약 1행 (usage_statement_item_id = NULL)

    usage_statement_items에서 (category_code, item_name)으로 DB ID를 조회한다.
    매칭 실패 시 usage_statement_item_id = NULL로 기록.
    """
    from src.repositories.db import get_connection

    try:
        with get_connection() as conn:
            # ── 항목 ID 맵 선조회 (1 query) ──────────────────────────────
            item_id_map = _fetch_item_id_map(conn, usage_statement_id)

            # ── 카테고리별 LLM 사유 미리 생성 ────────────────────────────
            cat_reasons = _compute_category_reasons(response)

            inserted = 0
            with conn.cursor() as cur:
                # ── 1. 항목별 행 ─────────────────────────────────────────
                for category_name, result in response.categories.items():
                    for item in result.items:
                        item_db_id  = _resolve_item_db_id(item_id_map, item)
                        result_code = _item_result_code(item, result)
                        reason      = _item_reason(item)
                        details     = _item_details(item, result)

                        cur.execute(_INSERT_AGENT_LOG_SQL, {
                            "project_id":         project_id,
                            "usage_statement_id": usage_statement_id,
                            "item_db_id":         item_db_id,
                            "result_code":        result_code,
                            "reason":             reason,
                            "details":            json.dumps(details, ensure_ascii=False),
                            "model_name":         model_name,
                        })
                        inserted += 1

                # ── 2. 사용내역서 전체 요약 행 (usage_statement_item_id = NULL) ──
                stmt_result_code, stmt_reason, stmt_details = _statement_summary(
                    response=response,
                    cat_reasons=cat_reasons,
                )
                cur.execute(_INSERT_AGENT_LOG_SQL, {
                    "project_id":         project_id,
                    "usage_statement_id": usage_statement_id,
                    "item_db_id":         None,
                    "result_code":        stmt_result_code,
                    "reason":             stmt_reason,
                    "details":            json.dumps(stmt_details, ensure_ascii=False),
                    "model_name":         model_name,
                })
                inserted += 1

        logger.info("[agent_log] legal INSERT %d건 완료 (항목 %d + 전체 1)", inserted, inserted - 1)
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
            if key not in result:   # 중복 시 첫 번째(작은 id) 우선
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
    return next((code for code, name in _CATEGORIES.items() if name == category), category)


def _item_result_code(item, result: "CategoryAuditResult") -> str:
    """
    항목 행 result_code 결정.

    우선순위:
      1. 항목 자체 불허          → hil
      2. 항목 자체 검토 필요      → hil
      3. 정상                    → success

    카테고리 한도 초과/공정률 부족은 사용내역서 전체 요약 행에서 기록한다.
    항목 자체가 적절한데 같은 카테고리의 다른 항목 때문에 hil로 오염되는 것을 막는다.
    """
    if item.allowed is False:
        return "hil"
    if getattr(item, "needs_human_review", False):
        return "hil"
    return "success"


def _item_reason(item) -> str:
    """Return the item-specific legal/LLM reason stored in agent_logs.reason."""
    reasoning = " ".join((getattr(item, "reasoning", "") or "").split()).strip()
    if reasoning:
        return reasoning[:1000]

    item_laws = list(getattr(item, "referenced_laws", []) or [])
    if item_laws:
        return f"{getattr(item, 'item', '해당 항목')}은(는) {', '.join(item_laws[:3])} 근거로 판단되었습니다."

    if getattr(item, "allowed", None) is False:
        return f"{getattr(item, 'item', '해당 항목')}은(는) 산안비 집행 항목으로 보기 어려워 보완 검토가 필요합니다."
    return f"{getattr(item, 'item', '해당 항목')}은(는) 산안비 집행 가능 항목으로 판단되었습니다."


def _compute_category_reasons(response: AuditResponse) -> dict[str, str]:
    """
    카테고리명 → LLM 생성 사유 맵.
    LLM 실패 시 _category_reason_fallback() 텍스트로 대체.
    """
    reasons: dict[str, str] = {}
    for category_name, result in response.categories.items():
        cat_code = next((k for k, v in _CATEGORIES.items() if v == category_name), category_name)
        sources = _build_cat_sources(
            category_name=category_name,
            category_code=cat_code,
            result=result,
            base_amount=response.base_amount,
        )
        llm_reason = _llm_synthesize_reason(
            category_name=category_name,
            result=result,
            base_amount=response.base_amount,
            sources=sources,
        )
        reasons[category_name] = llm_reason or _category_reason_fallback(result)
    return reasons


def _category_reason_fallback(result: "CategoryAuditResult") -> str:
    """LLM 실패 시 카테고리 사유 폴백 텍스트."""
    if result.exceeded and result.limit is not None:
        exceeded_amount = max(0.0, result.total - result.limit)
        return f"카테고리 한도 초과: {exceeded_amount:,.0f}원 초과"
    if result.usage_shortfall_amount and result.usage_shortfall_amount > 0:
        return f"공정률 기준 부족: {result.usage_shortfall_amount:,.0f}원 미달"
    disallowed = [it for it in result.items if it.allowed is False]
    if disallowed:
        return f"집행 불가 항목 포함: {', '.join(i.item for i in disallowed[:2])}"
    return "집행 가능"


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
                "detail":           "항목 reasoning 첫 문장",
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
    is_insufficient = bool(getattr(item, "needs_human_review", False))

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

    # ── detail: item.reasoning 첫 문장 (최대 150자) ───────────────────────────
    reasoning = " ".join((item.reasoning or "").split()).strip()
    first_sentence = reasoning.split(".")[0].strip()[:150]
    detail = first_sentence or None

    # ── confidence: 항목 confidence ───────────────────────────────────────────
    confidence = round(float(getattr(item, "confidence", 0.8)), 2)

    # ── category_code 조회 ───────────────────────────────────────────────────
    cat_code = _category_code_for(getattr(item, "category", ""))

    cat_entry: dict = {
        "status":           status,
        "supplement_codes": supplement_codes,
        "detail":           detail,
        "confidence":       confidence,
    }
    if basis:
        cat_entry["basis"] = basis

    # ── item section ──────────────────────────────────────────────────────────
    item_section: dict = {
        "item_name": item.item,
        "amount":    int(item.amount),
        "allowed":   item.allowed,
    }
    if item_laws:
        item_section["referenced_laws"] = item_laws
    judgment_source = getattr(item, "judgment_source", None)
    if judgment_source:
        item_section["judgment_source"] = judgment_source

    return {
        "categories": {cat_code: cat_entry} if cat_code else {},
        "item": item_section,
    }


def _statement_summary(
    *,
    response: AuditResponse,
    cat_reasons: dict[str, str],
) -> tuple[str, str, dict]:
    """
    사용내역서 전체 요약 행 (usage_statement_item_id = NULL) 을 위한
    (result_code, reason, details) 를 반환한다.

    result_code:
      - 이슈 항목 1건 이상 → hil
      - 전체 적절          → success

    details — V6 mockup 형식:
      Issues: {"result": "issues_found", "issues": [...]}
      Pass:   {"result": "pass", "summary": "..."}
    """
    all_issues: list[dict] = []

    for category_name, result in response.categories.items():
        # 항목 불허
        for item in result.items:
            if item.allowed is False:
                reasoning = (item.reasoning or "").strip()
                first = reasoning.split(".")[0].strip()
                all_issues.append({
                    "item": item.item,
                    "issue": first[:100] if first else "집행 불가 항목",
                })
        # 카테고리 한도 초과
        if result.exceeded and result.limit is not None:
            exceeded_amount = int(max(0.0, result.total - result.limit))
            all_issues.append({
                "item": category_name,
                "issue": "한도 초과",
                "exceeded_amount": exceeded_amount,
            })
        # 공정률 기준 미달
        if result.usage_shortfall_amount and result.usage_shortfall_amount > 0:
            all_issues.append({
                "item": category_name,
                "issue": "공정률 기준 미달",
                "shortfall_amount": int(result.usage_shortfall_amount),
            })

    result_code = "hil" if all_issues else "success"

    # ── reason: 부적절 카테고리의 LLM 사유를 결합 ──────────────────────────
    if result_code == "success":
        reason = "사용내역서 전체 항목이 법령 기준에 부합합니다."
    else:
        improper_cats = [
            name for name, r in response.categories.items()
            if r.status == "부적절"
        ]
        reason_parts = [
            cat_reasons[c] for c in improper_cats if cat_reasons.get(c)
        ]
        if reason_parts:
            reason = " ".join(reason_parts[:2])
        elif all_issues:
            reason = f"부적절 항목 포함: {', '.join(iss['item'] for iss in all_issues[:3])}"
        else:
            reason = "부적절 항목이 포함되어 있습니다."

    # ── details: V6 mockup 형식 ─────────────────────────────────────────────
    if result_code == "success":
        details: dict = {"result": "pass", "summary": "모든 항목 법령 기준 내 정산 적합"}
    else:
        details = {"result": "issues_found", "issues": all_issues}

    return result_code, reason, details


def _build_category_aggregates(response: AuditResponse) -> dict[str, dict]:
    """카테고리명 → 집계 수치 dict."""
    aggs: dict[str, dict] = {}
    for category_name, result in response.categories.items():
        agg: dict = {
            "status": result.status,
            "total":  int(result.total),
        }
        if result.limit is not None:
            agg["limit"]    = int(result.limit)
            agg["exceeded"] = result.exceeded
        if result.progress_rate is not None:
            agg["progress_rate"] = result.progress_rate
        if result.usage_shortfall_amount and result.usage_shortfall_amount > 0:
            agg["shortfall_amount"] = int(result.usage_shortfall_amount)
        aggs[category_name] = agg
    return aggs


# ── 카테고리 검증 실행 ────────────────────────────────────────────────────────

def _validate_blocks(base_amount: float, blocks, *, collection: str) -> AuditResponse:
    results: dict[str, CategoryAuditResult] = {}
    with ThreadPoolExecutor(max_workers=min(max(len(blocks), 1), 5)) as executor:
        futures = {
            executor.submit(_validate_category_block, block, collection): block.category_name
            for block in blocks
        }
        for future in as_completed(futures):
            category_name = futures[future]
            results[category_name] = future.result()
    return AuditResponse(base_amount=base_amount, categories=results)


def _validate_category_block(block, collection: str) -> CategoryAuditResult:
    retrieved = retrieve_category_context(block=block, collection=collection)
    rule_bundle = match_category_rules(block=block, retrieved=retrieved)
    computation = calculate_category_metrics(block=block, rule_bundle=rule_bundle)
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
