# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 함수 정의 ]
#
# 1. classify_intent()   : 사용자 질문을 6가지 intent로 분류 (Node 1)
# 2. retrieve_docs()     : Qdrant BM25+Vector Ensemble 검색 (Node 2)
# 3. fallback()          : 범위 외 질문에 고정 안내 메시지 반환 (Node 3)
# 4. grade_docs()        : 검색 문서 관련성 LLM 평가 및 필터링 (Node 4)
# 5. rewrite_query()     : 관련 문서 없을 시 검색 쿼리 재작성 (Node 5)
# 6. generate_answer()   : 관련 문서 기반 LLM 스트리밍 답변 생성 (Node 6)
# 7. route_by_intent()   : intent_classifier 이후 분기 조건 함수
# 8. route_after_grading(): doc_grader 이후 분기 조건 함수
#
# [ 설계 노트 ]
# - core/rag.py의 build_retriever()를 직접 재사용한다.
#   단, rewrite_query / rerank 는 AgenticRAGState 타입 불일치로
#   로직만 참고하여 ChatbotState 기반으로 재구현한다.
# - generate_answer()는 async 노드로 구현하며,
#   astream_events v2가 on_chat_model_stream 이벤트를 자동으로 캡처한다.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import src.core.llm_config as llm_config
from src.agents.chatbot_agent.prompts import (
    CHATBOT_ANSWER_PROMPT,
    CHATBOT_REWRITE_PROMPT,
    DOC_GRADE_PROMPT,
    FALLBACK_MESSAGE,
    NO_DOCS_FALLBACK_MESSAGE,
    INTENT_PROMPT,
)
from src.agents.chatbot_agent.state import ChatbotState
from src.core.rag import build_retriever
from src.core.storage import DEFAULT_COLLECTION, load_vectorstore

log = logging.getLogger(__name__)

MAX_RETRY = 3
MAX_MESSAGES = 20  # 메모리 누적 방지: 이 수를 초과하면 오래된 메시지부터 제거

# ── 답변 생성용 고성능 모델 (환경변수 우선, 없으면 gpt-4.1) ─────────────────
_ANSWER_MODEL = os.getenv("OPENAI_REPORT_MODEL", "gpt-4.1")


# ── structured_output 스키마 ─────────────────────────────────────────────────
class _IntentOutput(BaseModel):
    intent: str  # "카테고리판단" | "법령한도" | "적법성판단" | "계상기준" | "도급귀속" | "기타"


class _GradeOutput(BaseModel):
    relevance: str  # "relevant" | "irrelevant"


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _get_retriever():
    """벡터스토어 + BM25 앙상블 리트리버 반환 (싱글톤 캐시 활용)."""
    vectorstore = load_vectorstore(DEFAULT_COLLECTION)
    return build_retriever(vectorstore, DEFAULT_COLLECTION, k=8)


def _format_chat_history(messages: list) -> str:
    """messages 리스트를 프롬프트용 문자열로 변환."""
    if not messages:
        return "없음"
    lines = []
    for msg in messages[-6:]:  # 최근 3턴(6개 메시지)만 포함
        if isinstance(msg, HumanMessage):
            lines.append(f"사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"어시스턴트: {msg.content}")
    return "\n".join(lines) if lines else "없음"


def _format_docs(docs) -> str:
    """Document 리스트를 프롬프트용 문자열로 변환."""
    if not docs:
        return "관련 법령 근거 없음"
    return "\n\n---\n\n".join(
        f"[출처: {doc.metadata.get('source', doc.metadata.get('title', '알 수 없음'))}]\n{doc.page_content}"
        for doc in docs
    )


def _parse_json(raw: str) -> dict:
    """LLM 응답에서 JSON을 안전하게 파싱한다.

    LLM이 ```json ... ``` 코드블록으로 감싸는 경우를 처리한다.
    str.strip("```json")은 문자 집합을 제거하므로 사용하지 않는다.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw.strip())


def _extract_sources(docs) -> list[str]:
    """Document 메타데이터에서 출처 목록 추출."""
    sources = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("title") or doc.metadata.get("law_name")
        if src and src not in sources:
            sources.append(src)
    return sources


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(state: ChatbotState) -> dict:
    """사용자 질문을 6가지 intent로 분류한다.

    Returns:
        intent: "카테고리판단" | "법령한도" | "적법성판단" | "계상기준" | "도급귀속" | "기타"
        messages: 사용자 메시지 추가
    """
    question = state["question"]
    log.info(f"[intent_classifier] 질문: {question[:60]}")

    llm = llm_config.get()
    structured_llm = llm.with_structured_output(_IntentOutput)
    chain = INTENT_PROMPT | structured_llm

    chat_history = _format_chat_history(state.get("messages", []))

    try:
        result: _IntentOutput = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        intent = result.intent
    except Exception as e:
        log.warning(f"[intent_classifier] structured_output 실패 ({e}), 기타로 처리")
        intent = "기타"

    valid = {"카테고리판단", "법령한도", "적법성판단", "계상기준", "도급귀속", "기타"}
    if intent not in valid:
        intent = "기타"

    log.info(f"[intent_classifier] 분류 결과: {intent}")
    return {
        "intent": intent,
        "messages": [HumanMessage(content=question)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 — Retriever
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_docs(state: ChatbotState) -> dict:
    """Qdrant BM25+Vector Ensemble으로 관련 문서를 검색한다.

    Returns:
        retrieved_docs: 검색된 Document 리스트 (top-k=8)
    """
    question = state["question"]
    intent = state.get("intent", "")
    log.info(f"[retriever] intent={intent} | 질문: {question[:60]}")

    retriever = _get_retriever()
    docs = retriever.invoke(question)

    log.info(f"[retriever] 검색 결과: {len(docs)}건")
    return {"retrieved_docs": docs}


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 — Fallback Handler
# ══════════════════════════════════════════════════════════════════════════════

def fallback(state: ChatbotState) -> dict:
    """범위 외 질문이거나 근거 문서를 찾지 못한 경우 안내 메시지를 반환한다.

    - intent == "기타"              → 산안비 무관 질문 안내
    - graded_docs 비어있고 재시도 소진 → 근거 문서 없음 안내

    Returns:
        answer: 안내 메시지
        sources: 빈 리스트
        messages: AI 응답 메시지 추가
    """
    graded = state.get("graded_docs", [])
    retry  = state.get("retry_count", 0)
    intent = state.get("intent", "기타")

    # 재시도 소진 후 근거 문서 없음 → 별도 메시지
    if intent != "기타" and not graded and retry >= MAX_RETRY:
        log.info("[fallback] 근거 문서 없음 — 생성 차단")
        message = NO_DOCS_FALLBACK_MESSAGE
    else:
        log.info("[fallback] 범위 외 질문 처리")
        message = FALLBACK_MESSAGE

    return {
        "answer": message,
        "sources": [],
        "messages": [AIMessage(content=message)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 4 — Doc Grader
# ══════════════════════════════════════════════════════════════════════════════

def grade_docs(state: ChatbotState) -> dict:
    """검색된 각 문서의 관련성을 LLM으로 평가하고 필터링한다.

    hallucination의 주요 원인인 관련 없는 문서를 context에서 제거한다.

    Returns:
        graded_docs: 관련 있다고 판정된 Document 리스트
    """
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    log.info(f"[doc_grader] 평가 대상: {len(docs)}건")

    if not docs:
        log.info("[doc_grader] 검색 결과 없음 → graded_docs 빈 리스트")
        return {"graded_docs": []}

    llm = llm_config.get()
    structured_llm = llm.with_structured_output(_GradeOutput)
    chain = DOC_GRADE_PROMPT | structured_llm

    graded = []
    for doc in docs:
        try:
            result: _GradeOutput = chain.invoke({"question": question, "document": doc.page_content[:800]})
            if result.relevance == "relevant":
                graded.append(doc)
        except Exception as e:
            log.warning(f"[doc_grader] 평가 실패 ({e}), 해당 문서 제외")

    log.info(f"[doc_grader] 관련 문서: {len(graded)}건 / {len(docs)}건")
    return {"graded_docs": graded}


# ══════════════════════════════════════════════════════════════════════════════
# Node 5 — Rewrite Query
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query(state: ChatbotState) -> dict:
    """관련 문서가 없을 때 검색 성능 향상을 위해 질문을 재작성한다.

    core/rag.py의 rewrite_query 로직을 ChatbotState 기반으로 재구현.

    Returns:
        question: 재작성된 질문 (기존 question 덮어씌움)
        retry_count: 기존 값 + 1
    """
    original_question = state["question"]
    retry = state.get("retry_count", 0) + 1

    llm = llm_config.get()
    chain = CHATBOT_REWRITE_PROMPT | llm | StrOutputParser()

    try:
        new_question = chain.invoke({"question": original_question}).strip()
    except Exception as e:
        log.warning(f"[rewrite_query] 재작성 실패 ({e}), 원본 유지")
        new_question = original_question

    log.info(f"[rewrite_query] 재작성 ({retry}/{MAX_RETRY}): {new_question[:60]}")
    return {"question": new_question, "retry_count": retry}


# ══════════════════════════════════════════════════════════════════════════════
# Node 6 — Answer Generator
# ══════════════════════════════════════════════════════════════════════════════

async def generate_answer(state: ChatbotState) -> dict:
    """관련 문서를 context로 LLM 스트리밍 답변을 생성한다.

    async 노드로 구현하여 astream_events v2가
    on_chat_model_stream 이벤트를 통해 토큰 단위 스트리밍을 지원한다.

    Returns:
        answer: 생성된 전체 답변 텍스트
        sources: 참조한 법령/출처 목록
        messages: AI 응답 메시지 추가
    """
    question = state["question"]
    graded_docs = state.get("graded_docs", [])
    messages = state.get("messages", [])

    # 대화 길이 누적 방지: MAX_MESSAGES 초과 시 오래된 메시지 제거
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]
        log.info(f"[answer_generator] 메시지 트리밍: {MAX_MESSAGES}개로 축소")

    context = _format_docs(graded_docs)
    chat_history = _format_chat_history(messages)
    sources = _extract_sources(graded_docs)

    log.info(f"[answer_generator] 근거 문서: {len(graded_docs)}건 | 출처: {sources}")

    # 답변 생성은 고성능 모델 사용
    answer_llm = ChatOpenAI(
        model=_ANSWER_MODEL,
        temperature=0,
        streaming=True,
    )
    chain = CHATBOT_ANSWER_PROMPT | answer_llm | StrOutputParser()

    # astream()으로 토큰 단위 수집 → astream_events가 on_chat_model_stream 발생
    chunks = []
    async for chunk in chain.astream({
        "question": question,
        "context": context,
        "chat_history": chat_history,
    }):
        chunks.append(chunk)

    full_answer = "".join(chunks)
    log.info(f"[answer_generator] 답변 생성 완료 ({len(full_answer)}자)")

    return {
        "answer": full_answer,
        "sources": sources,
        "messages": [AIMessage(content=full_answer)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 라우팅 조건 함수
# ══════════════════════════════════════════════════════════════════════════════

def route_by_intent(
    state: ChatbotState,
) -> Literal["retrieve_docs", "fallback"]:
    """intent_classifier 이후 분기.

    - 기타 → fallback
    - 나머지 → retrieve_docs
    """
    intent = state.get("intent", "기타")
    if intent == "기타":
        log.info("[route] intent=기타 → fallback")
        return "fallback"
    log.info(f"[route] intent={intent} → retrieve_docs")
    return "retrieve_docs"


def route_after_grading(
    state: ChatbotState,
) -> Literal["rewrite_query", "generate_answer", "fallback"]:
    """doc_grader 이후 분기.

    - graded_docs 비어있고 retry < MAX_RETRY → rewrite_query
    - graded_docs 비어있고 retry >= MAX_RETRY → fallback (근거 없는 답변 생성 차단)
    - graded_docs 있음 → generate_answer
    """
    graded = state.get("graded_docs", [])
    retry = state.get("retry_count", 0)

    if not graded and retry < MAX_RETRY:
        log.info(f"[route] 관련 문서 없음 → rewrite_query (retry={retry})")
        return "rewrite_query"

    if not graded and retry >= MAX_RETRY:
        log.info(f"[route] 재시도 {MAX_RETRY}회 소진, 근거 문서 없음 → fallback")
        return "fallback"

    log.info(f"[route] 문서 {len(graded)}건 확보 → generate_answer")
    return "generate_answer"
