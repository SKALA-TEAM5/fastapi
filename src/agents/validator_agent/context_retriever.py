from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import load_vectorstore

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
    category_query = _build_category_query(block)
    category_docs = _retrieve_docs(question=category_query, collection=collection)

    item_docs: dict[str, list[Document]] = {}
    seed_docs = list(category_docs)
    for item in block.items:
        docs = _retrieve_docs(
            question=_build_item_query(block, item),
            collection=collection,
        )
        item_docs[item.item_name] = docs
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
    for doc in base_docs:
        heads |= set(_ARTICLE_PATTERN.findall(doc.page_content or ""))
    if not heads:
        return []

    vectorstore = load_vectorstore(collection_name=collection)
    full = vectorstore.get()
    expanded: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for page, meta in zip(full.get("documents", []), full.get("metadatas", [])):
        if any(head in page for head in heads):
            doc = Document(page_content=page, metadata=meta or {})
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                expanded.append(doc)
    return expanded


def _retrieve_docs(*, question: str, collection: str) -> list[Document]:
    vectorstore = load_vectorstore(collection_name=collection)
    retriever = build_retriever(vectorstore)
    state = {
        "question": question,
        "documents": [],
        "judgment": None,
        "retry_count": 0,
    }
    state = retrieve(state, retriever)
    state = rerank(state)

    while not state["documents"] and state.get("retry_count", 0) < MAX_RETRY:
        state = rewrite_query(state)
        state = retrieve(state, retriever)
        state = rerank(state)

    return state["documents"]


def _build_category_query(block: CategoryInputBlock) -> str:
    return (
        f"산업안전보건관리비 {block.category_name} 관련 법령 조항, 허용 규정, 사용불가 규정, 질의회시, "
        f"한도 규정, 공정률 기준"
    )


def _build_item_query(block: CategoryInputBlock, item: CategoryItemRow) -> str:
    extra = f" {item.remark}" if item.remark else ""
    return f"산업안전보건관리비 {block.category_name}에서 '{item.item_name}' 집행 가능 여부{extra}".strip()


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get("source", ""))
    return source, doc.page_content[:120]
