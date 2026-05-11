# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. build_retriever() : BM25 + 벡터 앙상블 리트리버 생성
# 2. retrieve()        : LangGraph 노드 — 문서 검색 수행
# 3. rerank()          : Cross-encoder 기반 검색 결과 재순위
# 4. rewrite_query()   : LLM 기반 검색 쿼리 재작성
# --------------------------------------------------------------------------
import logging
from typing import NamedTuple

from kiwipiepy import Kiwi
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

import src.core.llm_config as llm_config
from src.prompts import REWRITE_PROMPT
from src.schemas.shared import AgenticRAGState
from src.core.storage import load_collection_documents

log = logging.getLogger(__name__)

MAX_RETRY = 3

_kiwi = Kiwi()
_RERANK_MODEL = "BAAI/bge-reranker-base"
_TOP_N = 5
_BATCH_SIZE = 8


class _RetrieverCache(NamedTuple):
    fingerprint: tuple
    corpus: list[Document]
    bm25: BM25Retriever


_retriever_cache: _RetrieverCache | None = None
_rerank_model: HuggingFaceCrossEncoder | None = None


def rewrite_query(state: AgenticRAGState) -> AgenticRAGState:
    """관련 문서가 없을 때 검색 성능 향상을 위해 질문을 재작성."""
    llm = llm_config.get()
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    new_q = rewrite_chain.invoke({"question": state["question"]})
    retry = state.get("retry_count", 0) + 1
    log.info(f"쿼리 재작성 ({retry}/{MAX_RETRY}): {new_q[:60]}...")
    return {**state, "question": new_q, "retry_count": retry}


def _tokenize(text: str) -> list[str]:
    return [token.form for token in _kiwi.tokenize(text)]


def _get_corpus_and_bm25(
    vectorstore,
    collection_name: str,
    use_pdf_only: bool,
    k: int,
) -> tuple[list[Document], BM25Retriever]:
    global _retriever_cache

    corpus = load_collection_documents(
        collection_name,
        source_exclude="web_법령" if use_pdf_only else None,
    )
    fingerprint = (collection_name, use_pdf_only, len(corpus))

    if _retriever_cache and _retriever_cache.fingerprint == fingerprint:
        log.info("Retriever 캐시 적중 — corpus/BM25 재구축 스킵")
        _retriever_cache.bm25.k = k
        return _retriever_cache.corpus, _retriever_cache.bm25

    log.info(f"BM25 인덱스 구축 중 (문서 {len(corpus)}개)")
    bm25 = BM25Retriever.from_documents(corpus, preprocess_func=_tokenize, k=k)
    _retriever_cache = _RetrieverCache(fingerprint=fingerprint, corpus=corpus, bm25=bm25)
    return corpus, bm25


def build_retriever(vectorstore, collection_name: str, k: int = 8) -> EnsembleRetriever:
    pdf_corpus = load_collection_documents(collection_name, source_exclude="web_법령")
    use_pdf_only = len(pdf_corpus) > 0

    _, bm25_retriever = _get_corpus_and_bm25(vectorstore, collection_name, use_pdf_only, k)

    if use_pdf_only:
        log.info(f"PDF 청크 {len(pdf_corpus)}개로 Ensemble 구성")
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    else:
        log.info("PDF 청크 없음 — 웹 전체로 Ensemble 구성")
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def retrieve(state: AgenticRAGState, retriever: BaseRetriever) -> AgenticRAGState:
    docs = retriever.invoke(state["question"])
    return {**state, "retrieved_docs": docs}


def _get_rerank_model() -> HuggingFaceCrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        log.info(f"ReRanker 로드 중 ({device}): {_RERANK_MODEL}")
        _rerank_model = HuggingFaceCrossEncoder(
            model_name=_RERANK_MODEL,
            model_kwargs={"device": device},
        )
    return _rerank_model


def _score_in_batches(model: HuggingFaceCrossEncoder, pairs: list) -> list[float]:
    scores: list[float] = []
    for i in range(0, len(pairs), _BATCH_SIZE):
        batch = pairs[i : i + _BATCH_SIZE]
        scores.extend(model.score(batch))
    return scores


def rerank(state: AgenticRAGState) -> AgenticRAGState:
    docs = state["retrieved_docs"]
    if not docs:
        return state

    model = _get_rerank_model()
    query = state["question"]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _score_in_batches(model, pairs)

    ranked: list[tuple[float, Document]] = sorted(
        zip(scores, docs), key=lambda x: x[0], reverse=True
    )
    top_docs = [doc for _, doc in ranked[:_TOP_N]]

    log.info(f"ReRanker: {len(docs)}개 → {len(top_docs)}개 (상위 점수: {ranked[0][0]:.3f})")
    return {**state, "retrieved_docs": top_docs}
