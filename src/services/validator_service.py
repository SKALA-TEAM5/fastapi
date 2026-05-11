# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. run_audit_service() : 카테고리 묶음 검토 서비스
# 2. validate_document_service() : 단일 카테고리 검토 서비스
# 3. validate_usage_statement_service() : 사용내역서 검토 서비스
# --------------------------------------------------------------------------
from __future__ import annotations

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
from src.agents.validator_agent.rule_matcher import match_category_rules
from src.core.storage import DEFAULT_COLLECTION
from src.schemas.validator import AuditResponse, CategoryAuditResult


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
) -> AuditResponse:
    parsed = parse_usage_statement(document)
    return _validate_blocks(parsed.base_amount, parsed.blocks, collection=collection)


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
