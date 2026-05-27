# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-05-26
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. CategoryRetrievedContext    : 카테고리 RAG 검색 결과 집계 데이터 클래스
# 2. retrieve_category_context() : 카테고리 + 항목별 병렬 벡터 DB 검색
# 3. _build_category_query()     : 카테고리 수준 검색 쿼리 생성 (한도/전체 규정 중심)
# 4. _build_item_query()         : 항목 수준 검색 쿼리 생성
# --------------------------------------------------------------------------
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain_core.documents import Document

from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import DEFAULT_COLLECTION, load_vectorstore
from src.repositories import LegalRulesRepository


@dataclass
class CategoryRetrievedContext:
    category_docs: list[Document]
    item_docs: dict[str, list[Document]]
    exception_docs: list[Document]

    @property
    def all_docs(self) -> list[Document]:
        merged: list[Document] = []
        for doc in self.category_docs + self.exception_docs:
            if _doc_key(doc) not in {_doc_key(existing) for existing in merged}:
                merged.append(doc)
        for docs in self.item_docs.values():
            for doc in docs:
                if _doc_key(doc) not in {_doc_key(existing) for existing in merged}:
                    merged.append(doc)
        return merged


def retrieve_category_context(
    *,
    block: CategoryInputBlock,
    collection: str,
) -> CategoryRetrievedContext:
    repo = LegalRulesRepository()
    vectorstore = load_vectorstore(collection_name=collection)
    retriever = build_retriever(vectorstore, collection_name=collection)
    category_query = _build_category_query(block, repo=repo)
    category_docs = _retrieve_docs(question=category_query, retriever=retriever)

    item_docs: dict[str, list[Document]] = {}
    with ThreadPoolExecutor(max_workers=min(max(len(block.items), 1), 4)) as executor:
        futures = {
            executor.submit(
                _retrieve_docs,
                question=_build_item_query(block, item, repo=repo),
                retriever=retriever,
            ): item.item_name
            for item in block.items
        }
        for future in as_completed(futures):
            item_name = futures[future]
            docs = future.result()
            item_docs[item_name] = docs

    # VectorDB 1차 검색 결과가 없는 항목 → 더 넓은 fallback query로 재시도
    for item in block.items:
        if not item_docs.get(item.item_name):
            fallback_query = _build_item_fallback_query(block, item)
            fallback_docs = _retrieve_docs(question=fallback_query, retriever=retriever)
            if fallback_docs:
                item_docs[item.item_name] = fallback_docs

    # expand_exception_context 제거 — Qdrant 전체 문서 전수 스캔으로 인한 성능 병목
    # 예외/단서 조항은 항목별 Qdrant 검색 결과에 이미 포함되거나 LLM fallback에서 처리
    return CategoryRetrievedContext(
        category_docs=category_docs,
        item_docs=item_docs,
        exception_docs=[],
    )


def _retrieve_docs(*, question: str, retriever) -> list[Document]:
    state = {
        "question": question,
        "retrieved_docs": [],
        "judgment": None,
        "retry_count": 0,
    }
    state = retrieve(state, retriever)
    state = rerank(state)

    while not state["retrieved_docs"] and state.get("retry_count", 0) < MAX_RETRY:
        state = rewrite_query(state)
        state = retrieve(state, retriever)
        state = rerank(state)

    return state["retrieved_docs"]


def _build_category_query(block: CategoryInputBlock, *, repo: LegalRulesRepository) -> str:
    hint_terms: list[str] = [block.category_code, block.category_name]
    law_terms: list[str] = []
    evidence_terms: list[str] = []

    limit_pct, limit_rule, limit_laws = repo.find_category_limit(block.category_name)
    if limit_pct is not None:
        hint_terms.append(str(limit_pct))
    if limit_rule:
        evidence_terms.append(limit_rule)
    law_terms.extend(limit_laws)

    # items 루프 제거 — find_validator_matches는 항목별 쿼리(_build_item_query)에서 이미 호출됨
    # 카테고리 쿼리는 한도/전체 규정 문서 검색에만 집중

    category_hint = ", ".join(dict.fromkeys(term.strip() for term in hint_terms if term and term.strip()))
    law_hint = ", ".join(dict.fromkeys(term.strip() for term in law_terms if term and term.strip()))
    evidence_hint = ", ".join(dict.fromkeys(term.strip() for term in evidence_terms if term and term.strip()))
    return " | ".join(
        part for part in [
            f"산업안전보건관리비 카테고리 '{block.category_name}' 검토",
            f"카테고리 코드 {block.category_code}" if block.category_code else "",
            f"우선 검토 규칙 {category_hint}" if category_hint else "",
            f"관련 조항 {law_hint}" if law_hint else "",
            f"한도 및 예외 기준 {evidence_hint}" if evidence_hint else "",
            "허용 규정, 사용불가 규정, 질의회시, 한도 규정, 공정률 기준",
        ] if part
    ).strip()


def _build_item_query(
    block: CategoryInputBlock,
    item: CategoryItemRow,
    *,
    repo: LegalRulesRepository,
) -> str:
    extra = f" {item.remark}" if item.remark else ""
    item_text_norm = item.item_name.lower()
    prelim_matches = repo.find_validator_matches(
        category=block.category_name,
        item_text=item.item_name,
        retrieved_context="",
        limit=4,
    )
    hint_terms: list[str] = [block.category_code, block.category_name, item.item_name]
    law_terms: list[str] = []
    evidence_terms: list[str] = []

    for match in prelim_matches:
        # allowed=True인 규칙의 조항/근거만 힌트로 사용
        # allowed=False(disallowed) 규칙의 evidence를 넣으면
        # VectorDB가 잘못된 카테고리 법령 청크를 불러와 LLM을 오도함
        if match.allowed is True:
            law_terms.extend(match.referenced_laws)
            if match.evidence:
                evidence_terms.append(match.evidence)

    category_hint = ", ".join(dict.fromkeys(term.strip() for term in hint_terms if term and term.strip()))
    law_hint = ", ".join(dict.fromkeys(term.strip() for term in law_terms if term and term.strip()))
    evidence_hint = ", ".join(dict.fromkeys(term.strip() for term in evidence_terms if term and term.strip()))

    return " | ".join(
        part for part in [
            f"산업안전보건관리비 '{block.category_name}'에서 '{item.item_name}' 집행 가능 여부{extra}".strip(),
            f"카테고리 힌트 {category_hint}" if category_hint else "",
            f"관련 조항 {law_hint}" if law_hint else "",
            f"관련 근거 {evidence_hint}" if evidence_hint else "",
        ] if part
    ).strip()


def _build_item_fallback_query(block: CategoryInputBlock, item: CategoryItemRow) -> str:
    """
    1차 VectorDB 검색에서 결과가 없을 때 사용하는 넓은 법령 검색 쿼리.
    카테고리·항목명에 의존하지 않고 법령 취지 중심으로 검색.
    """
    return (
        f"{item.item_name} 산업안전보건관리비 사용 허용 여부 "
        f"건설업 산업안전보건관리비 계상 및 사용기준 제7조 산업재해예방 법령 근거"
    )


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get("source", ""))
    return source, doc.page_content[:120]
