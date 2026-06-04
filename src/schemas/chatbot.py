# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
#
# [ 주요 클래스 정의 ]
#
# 1. ChatRequest  : POST /api/v1/chat 요청 스키마
# 2. ChatEvent    : SSE 스트리밍 이벤트 단위 스키마
#
# [ SSE 이벤트 타입 ]
# - session_id : 세션 ID 확정 (신규 생성 또는 그대로 반환)
# - intent     : 질문 의도 분류 결과
# - token      : LLM 토큰 단위 스트리밍
# - sources    : 참조한 법령/출처 목록
# - error      : 오류 발생 시 안내 메시지
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/v1/chat 요청 바디."""

    question: str = Field(..., description="사용자 질문", min_length=1)
    session_id: Optional[str] = Field(
        default=None,
        description="대화 세션 ID. 미전달 시 서버에서 UUID를 생성하고 session_id 이벤트로 반환한다.",
    )

    model_config = {"json_schema_extra": {"example": {"question": "안전모는 몇 번 카테고리인가요?", "session_id": None}}}


class ChatEvent(BaseModel):
    """SSE 스트리밍 이벤트 단위 스키마.

    프론트엔드에서 `data` 필드를 JSON 파싱하여 type별로 처리한다.
    """

    type: str = Field(description="이벤트 타입: session_id | intent | token | sources | error")
    value: Any = Field(description="이벤트 값. 타입에 따라 str 또는 list[str]")
