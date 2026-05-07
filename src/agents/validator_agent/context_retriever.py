from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import DEFAULT_COLLECTION, load_collection_documents, load_vectorstore
from src.repositories import LegalRulesRepository

_EXCEPTION_KEYWORDS = ("단,", "다만", "예외", "제외", "불가")
_ARTICLE_PATTERN = re.compile(r"제\d+조(?:제\d+항)?(?:제\d+호)?|별표\s*\d+")


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
    seed_docs = list(category_docs)
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
            seed_docs.extend(docs)

    exception_docs = expand_exception_context(base_docs=seed_docs, collection=collection)
    return CategoryRetrievedContext(
        category_docs=category_docs,
        item_docs=item_docs,
        exception_docs=exception_docs,
    )


def expand_exception_context(*, base_docs: list[Document], collection: str) -> list[Document]:
    if not base_docs:
        return []

    need_expand = any(
        any(keyword in (doc.page_content or "") for keyword in _EXCEPTION_KEYWORDS)
        for doc in base_docs
    )
    if not need_expand:
        return []

    heads = set()
    preferred_sources = {
        str(doc.metadata.get("source", "")).strip()
        for doc in base_docs
        if str(doc.metadata.get("source", "")).strip()
    }
    for doc in base_docs:
        heads |= set(_ARTICLE_PATTERN.findall(doc.page_content or ""))
    if not heads:
        return []

    expanded: list[Document] = []
    seen: set[tuple[str, str]] = set()
    per_head_count: dict[str, int] = {}
    for doc in load_collection_documents(collection):
        source = str(doc.metadata.get("source", "")).strip()
        if preferred_sources and source and source not in preferred_sources:
            continue
        matched_heads = [head for head in heads if head in (doc.page_content or "")]
        if not matched_heads:
            continue
        primary_head = matched_heads[0]
        if per_head_count.get(primary_head, 0) >= 6:
            continue
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            expanded.append(doc)
            per_head_count[primary_head] = per_head_count.get(primary_head, 0) + 1
    return expanded


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

    for item in block.items[:3]:
        prelim_matches = repo.find_validator_matches(
            category=block.category_name,
            item_text=item.item_name,
            retrieved_context="",
            limit=3,
        )
        for match in prelim_matches:
            hint_terms.append(match.rule_type)
            law_terms.extend(match.referenced_laws)
            if match.evidence:
                evidence_terms.append(match.evidence)

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
    allow_terms: list[str] = []
    disallow_terms: list[str] = []
    law_terms: list[str] = []
    evidence_terms: list[str] = []
    profile = repo.validator_profiles.get(block.category_code or "", {})
    allow_hits = [term for term in profile.get("allow_terms", []) if str(term).lower() in item_text_norm]
    disallow_hits = [term for term in profile.get("disallow_terms", []) if str(term).lower() in item_text_norm]

    if allow_hits:
        allow_terms.extend(allow_hits[:3])
    if disallow_hits:
        disallow_terms.extend(disallow_hits[:3])

    for match in prelim_matches:
        hint_terms.append(match.rule_type)
        law_terms.extend(match.referenced_laws)
        if match.evidence:
            evidence_terms.append(match.evidence)

    category_hint = ", ".join(dict.fromkeys(term.strip() for term in hint_terms if term and term.strip()))
    law_hint = ", ".join(dict.fromkeys(term.strip() for term in law_terms if term and term.strip()))
    allow_hint = ", ".join(dict.fromkeys(term.strip() for term in allow_terms if term and term.strip()))
    disallow_hint = ", ".join(dict.fromkeys(term.strip() for term in disallow_terms if term and term.strip()))
    evidence_hint = ", ".join(dict.fromkeys(term.strip() for term in evidence_terms if term and term.strip()))

    return " | ".join(
        part for part in [
            f"산업안전보건관리비 '{block.category_name}'에서 '{item.item_name}' 집행 가능 여부{extra}".strip(),
            f"카테고리 힌트 {category_hint}" if category_hint else "",
            f"허용 범위 키워드 {allow_hint}" if allow_hint else "",
            f"사용불가 기준 키워드 {disallow_hint}" if disallow_hint else "",
            f"관련 조항 {law_hint}" if law_hint else "",
            f"관련 근거 {evidence_hint}" if evidence_hint else "",
        ] if part
    ).strip()


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get("source", ""))
    return source, doc.page_content[:120]
