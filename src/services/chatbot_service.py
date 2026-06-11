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
# - on_chain_start (노드 진입)       → {"type": "status",  "value": "...중..."}
# - on_chain_end (intent_classifier) → {"type": "intent",  "value": intent}
# - on_chat_model_stream             → {"type": "token",   "value": chunk}
# - on_chain_end (answer_generator)  → {"type": "sources", "value": [...]}
# - on_chain_end (fallback_handler)  → {"type": "token",   "value": message}
#
# [ 안전 장치 ]
# 1. 동시 요청 중복 방지  : session_id별 asyncio.Lock
# 2. 대화 길이 누적 방지  : nodes.py MAX_MESSAGES (generate_answer 내부)
# 3. 스트리밍 타임아웃    : 전체 120s / 토큰 간 30s
# 4. 세션 만료 안내       : MemorySaver에 thread 없으면 session_reset 이벤트
# 5. 질문 전처리          : schemas/chatbot.py ChatRequest.sanitize_question
# --------------------------------------------------------------------------
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from src.agents.chatbot_agent.agent import get_compiled_graph
from src.repositories.orchestrator_repository import insert_agent_usage_record

log = logging.getLogger(__name__)

_DONE_SIGNAL = "data: [DONE]\n\n"

# 타임아웃 설정
_STREAM_TOTAL_TIMEOUT  = 120  # 전체 스트리밍 최대 시간 (초)
_STREAM_TOKEN_TIMEOUT  = 30   # 토큰 간 최대 대기 시간 (초)

# 세션별 Lock (동시 요청 중복 방지)
_session_locks: dict[str, asyncio.Lock] = {}

# 노드 진입 시 프론트에 표시할 상태 메시지
_NODE_STATUS: dict[str, str] = {
    "intent_classifier": "질문 유형 분석 중...",
    "retriever":         "관련 법령 검색 중...",
    "doc_grader":        "문서 관련성 검토 중...",
    "rewrite_query":     "질문 재구성 중...",
    "answer_generator":  "답변 생성 중...",
    "fallback_handler":  "답변 준비 중...",
}


def _sse(type_: str, value) -> str:
    """SSE 포맷 문자열 생성."""
    payload = json.dumps({"type": type_, "value": value}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """세션별 Lock을 가져온다. 없으면 생성."""
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


def _has_session_state(session_id: str) -> bool:
    """MemorySaver에 해당 session_id(thread_id)의 상태가 존재하는지 확인."""
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        return bool(state and state.values)
    except Exception:
        return False


async def _stream_events(
    graph,
    initial_state: dict,
    config: dict,
    usage: dict,
) -> AsyncGenerator[str, None]:
    """astream_events를 소비하며 SSE 이벤트를 yield한다.

    - 전체 타임아웃: _STREAM_TOTAL_TIMEOUT 초
    - 토큰 간 타임아웃: _STREAM_TOKEN_TIMEOUT 초
    - usage: 토큰 누적용 mutable dict {"input_tokens": int, "output_tokens": int, "model_name": str | None}
    """
    async def _next_event(ait):
        return await ait.__anext__()

    ait = graph.astream_events(initial_state, config=config, version="v2").__aiter__()
    total_deadline = asyncio.get_event_loop().time() + _STREAM_TOTAL_TIMEOUT

    while True:
        remaining = total_deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError("전체 스트리밍 시간 초과")

        try:
            event = await asyncio.wait_for(
                _next_event(ait),
                timeout=min(_STREAM_TOKEN_TIMEOUT, remaining),
            )
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("응답 대기 시간 초과")

        ev_name = event.get("name", "")
        ev_type = event.get("event", "")
        data    = event.get("data", {})

        # 노드 진입 시 상태 메시지 전송
        if ev_type == "on_chain_start" and ev_name in _NODE_STATUS:
            yield _sse("status", _NODE_STATUS[ev_name])

        # intent 분류 결과 전송
        elif ev_type == "on_chain_end" and ev_name == "intent_classifier":
            output = data.get("output", {})
            intent = output.get("intent", "")
            if intent:
                yield _sse("intent", intent)

        # LLM 토큰 단위 스트리밍 — answer_generator 노드에서 발생한 것만 통과
        elif ev_type == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node != "answer_generator":
                continue
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield _sse("token", chunk.content)

        # fallback 안내 메시지 전송
        elif ev_type == "on_chain_end" and ev_name == "fallback_handler":
            output = data.get("output", {})
            answer = output.get("answer", "")
            if answer:
                yield _sse("token", answer)

        # 참조 법령 출처 전송
        elif ev_type == "on_chain_end" and ev_name == "answer_generator":
            output = data.get("output", {})
            sources = output.get("sources", [])
            if sources:
                yield _sse("sources", sources)

        # LLM 호출 완료 시 토큰 누적
        elif ev_type == "on_chat_model_end":
            output = data.get("output", {})
            meta = output.usage_metadata if hasattr(output, "usage_metadata") else {}
            if meta:
                usage["input_tokens"]  += meta.get("input_tokens", 0)
                usage["output_tokens"] += meta.get("output_tokens", 0)
            if not usage["model_name"]:
                resp_meta = output.response_metadata if hasattr(output, "response_metadata") else {}
                usage["model_name"] = resp_meta.get("model_name")


async def stream_chat(
    question: str,
    session_id: str | None = None,
    user_id: int | None = None,
) -> AsyncGenerator[str, None]:
    """챗봇 그래프를 실행하고 SSE 이벤트를 스트리밍한다.

    Args:
        question   : 사용자 질문 (schemas/chatbot.py에서 전처리 완료)
        session_id : 대화 세션 ID (없으면 UUID 생성)

    Yields:
        SSE 포맷 문자열 (``data: {...}\\n\\n`` 형태)
    """
    # ── 1. session_id 확정 ────────────────────────────────────────────────────
    is_new_session = session_id is None
    session_id = session_id or str(uuid.uuid4())

    # ── 2. 동시 요청 중복 방지 ────────────────────────────────────────────────
    lock = _get_session_lock(session_id)
    if lock.locked():
        log.warning(f"[chatbot_service] 중복 요청 차단: {session_id}")
        yield _sse("error", "이전 답변을 생성 중입니다. 잠시 후 다시 시도해 주세요.")
        yield _DONE_SIGNAL
        return

    async with lock:
        # ── 3. session_id 이벤트 전송 ─────────────────────────────────────────
        if is_new_session:
            log.info(f"[chatbot_service] 새 세션 생성: {session_id}")
        else:
            log.info(f"[chatbot_service] 기존 세션 연결: {session_id}")

        yield _sse("session_id", session_id)

        # ── 4. 세션 만료 감지 ─────────────────────────────────────────────────
        if not is_new_session and not _has_session_state(session_id):
            log.info(f"[chatbot_service] 세션 만료 감지: {session_id}")
            yield _sse("session_reset", "이전 대화 기록이 만료되었습니다. 새 대화를 시작합니다.")

        # ── 5. 그래프 실행 ────────────────────────────────────────────────────
        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "question":       question,
            "intent":         "",
            "retrieved_docs": [],
            "graded_docs":    [],
            "retry_count":    0,
            "sources":        [],
            "answer":         "",
            "messages":       [],
        }

        usage: dict = {"input_tokens": 0, "output_tokens": 0, "model_name": None}

        try:
            async for sse_event in _stream_events(graph, initial_state, config, usage):
                yield sse_event

        except asyncio.TimeoutError as e:
            log.error(f"[chatbot_service] 타임아웃: {e}")
            yield _sse("error", "응답 시간이 초과되었습니다. 다시 시도해 주세요.")

        except Exception as e:
            log.error(f"[chatbot_service] 스트리밍 오류: {e}", exc_info=True)
            yield _sse("error", "일시적인 오류가 발생했습니다. 다시 시도해 주세요.")

        # 토큰이 수집됐고 user_id가 있으면 agent_usage_records에 기록 (fire-and-forget)
        if user_id is not None and (usage["input_tokens"] > 0 or usage["output_tokens"] > 0):
            asyncio.create_task(
                asyncio.to_thread(
                    insert_agent_usage_record,
                    project_id=999,
                    usage_statement_id=None,
                    agent_type_code="chatbot",
                    model_name=usage["model_name"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    requested_by_user_id=user_id,
                )
            )

        yield _DONE_SIGNAL
