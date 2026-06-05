# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 함수 정의 ]
#
# 1. build_chatbot_graph() : LangGraph StateGraph 조립 및 반환
# 2. get_compiled_graph()  : MemorySaver checkpointer가 연결된 컴파일 그래프 반환
#                            (싱글톤 — 앱 시작 시 1회 초기화, 재사용)
#
# [ 그래프 흐름 ]
#   intent_classifier
#       ├── 기타         → fallback_handler → END
#       └── 그 외        → retriever → doc_grader
#                                          ├── 관련 없음 & retry<3 → rewrite_query → retriever
#                                          └── 관련 있음 or retry>=3 → answer_generator → END
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from threading import Lock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.chatbot_agent.nodes import (
    classify_intent,
    fallback,
    generate_answer,
    grade_docs,
    retrieve_docs,
    rewrite_query,
    route_after_grading,
    route_by_intent,
)
from src.agents.chatbot_agent.state import ChatbotState

log = logging.getLogger(__name__)

# ── 싱글톤 ──────────────────────────────────────────────────────────────────
_compiled_graph = None
_graph_lock = Lock()


def build_chatbot_graph() -> StateGraph:
    """LangGraph StateGraph를 조립하고 반환한다. (미컴파일 상태)"""
    graph = StateGraph(ChatbotState)

    # 노드 등록
    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("retriever",         retrieve_docs)
    graph.add_node("fallback_handler",  fallback)
    graph.add_node("doc_grader",        grade_docs)
    graph.add_node("rewrite_query",     rewrite_query)
    graph.add_node("answer_generator",  generate_answer)

    # 진입점
    graph.set_entry_point("intent_classifier")

    # intent_classifier 이후 분기
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "retrieve_docs": "retriever",
            "fallback":      "fallback_handler",
        },
    )

    # retriever → doc_grader (항상)
    graph.add_edge("retriever", "doc_grader")

    # doc_grader 이후 분기
    graph.add_conditional_edges(
        "doc_grader",
        route_after_grading,
        {
            "rewrite_query":   "rewrite_query",
            "generate_answer": "answer_generator",
            "fallback":        "fallback_handler",  # 근거 문서 없으면 생성 차단
        },
    )

    # rewrite → 다시 retriever (루프백)
    graph.add_edge("rewrite_query", "retriever")

    # 종료 엣지
    graph.add_edge("fallback_handler",  END)
    graph.add_edge("answer_generator",  END)

    return graph


def get_compiled_graph():
    """MemorySaver checkpointer가 연결된 컴파일 그래프를 반환한다.

    앱 생명주기 동안 단 1회 초기화되며 이후 재사용된다.
    thread_id(=session_id)를 기준으로 대화 상태를 분리 보관한다.
    서버 재시작 또는 탭을 닫으면 대화 기록이 소멸한다.
    """
    global _compiled_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:
                log.info("ChatbotGraph 초기화 (MemorySaver checkpointer)")
                graph = build_chatbot_graph()
                checkpointer = MemorySaver()
                _compiled_graph = graph.compile(checkpointer=checkpointer)
    return _compiled_graph
