# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 클래스 정의 ]
#
# 1. ChatbotState : 챗봇 LangGraph 워크플로우 공유 상태 스키마
#
# [ 설계 노트 ]
# - messages 필드는 Annotated[list, operator.add] 로 선언해야
#   MemorySaver checkpointer가 대화 기록을 덮어쓰지 않고 누적한다.
# - retrieved_docs / graded_docs 는 매 턴마다 초기화되므로 누적 불필요.
# --------------------------------------------------------------------------
from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class ChatbotState(MessagesState):
    """LangGraph 그래프 전체에서 공유되는 챗봇 상태.

    Attributes:
        question        현재 사용자 질문 (rewrite 시 덮어씌워짐)
        intent          intent_classifier 분류 결과
        retrieved_docs  retriever 노드가 반환한 원본 문서 목록
        graded_docs     doc_grader 노드가 필터링한 관련 문서 목록
        retry_count     rewrite_query 재시도 횟수 (최대 MAX_RETRY)
        sources         answer_generator가 참조한 법령/출처 목록
        answer          최종 생성 답변 (스트리밍 완료 후 전체 텍스트)
        messages        대화 기록 — MessagesState 상속으로 자동 누적
    """

    question: str
    intent: str
    retrieved_docs: List[Document]
    graded_docs: List[Document]
    retry_count: int
    sources: List[str]
    answer: str
    # messages 는 MessagesState(Annotated[list, add_messages])에서 상속
