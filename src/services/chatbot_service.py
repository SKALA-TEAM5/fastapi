# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 함수 정의 ]
#
# 1. stream_chat() : LangGraph astream_events를 실행하고
#                    SSE 포맷 이벤트를 async generator로 반환한다.
#
# [ SSE 이벤트 변환 규칙 ]
# - on_chain_end (intent_classifier) → {"type": "intent", "value": intent}
# - on_chat_model_stream             → {"type": "token", "value": chunk}
# - on_chain_end (answer_generator)  → {"type": "sources", "value": [...]}
# - on_chain_end (fallback_handler)  → {"type": "token", "value": message}
# --------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

from src.agents.chatbot_agent.agent import get_compiled_graph

log = logging.getLogger(__name__)

_DONE_SIGNAL = "data: [DONE]\n\n"


def _sse(type_: str, value) -> str:
    """SSE 포맷 문자열 생성."""
    payload = json.dumps({"type": type_, "value": value}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def stream_chat(
    question: str,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """챗봇 그래프를 실행하고 SSE 이벤트를 스트리밍한다.

    Args:
        question   : 사용자 질문
        session_id : 대화 세션 ID (없으면 UUID 생성)

    Yields:
        SSE 포맷 문자열 (``data: {...}\\n\\n`` 형태)
    """
    # session_id 확정
    is_new_session = session_id is None
    session_id = session_id or str(uuid.uuid4())

    if is_new_session:
        log.info(f"[chatbot_service] 새 세션 생성: {session_id}")
        yield _sse("session_id", session_id)
    else:
        log.info(f"[chatbot_service] 기존 세션 연결: {session_id}")
        yield _sse("session_id", session_id)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "question":      question,
        "intent":        "",
        "retrieved_docs": [],
        "graded_docs":   [],
        "retry_count":   0,
        "sources":       [],
        "answer":        "",
        "messages":      [],
    }

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            ev_name = event.get("name", "")
            ev_type = event.get("event", "")
            data    = event.get("data", {})

            # intent 분류 결과 전송
            if ev_type == "on_chain_end" and ev_name == "intent_classifier":
                output = data.get("output", {})
                intent = output.get("intent", "")
                if intent:
                    yield _sse("intent", intent)

            # LLM 토큰 단위 스트리밍 — answer_generator 노드에서 발생한 것만 통과
            # (intent_classifier, doc_grader의 structured_output 토큰 차단)
            elif ev_type == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node != "answer_generator":
                    continue
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield _sse("token", chunk.content)

            # fallback 안내 메시지 전송 (LLM 없이 고정 메시지)
            elif ev_type == "on_chain_end" and ev_name == "fallback_handler":
                output = data.get("output", {})
                answer = output.get("answer", "")
                if answer:
                    yield _sse("token", answer)

            # 참조 법령 출처 전송 (answer_generator 완료 시)
            elif ev_type == "on_chain_end" and ev_name == "answer_generator":
                output = data.get("output", {})
                sources = output.get("sources", [])
                if sources:
                    yield _sse("sources", sources)

    except Exception as e:
        log.error(f"[chatbot_service] 스트리밍 오류: {e}", exc_info=True)
        yield _sse("error", "일시적인 오류가 발생했습니다. 다시 시도해 주세요.")

    yield _DONE_SIGNAL
