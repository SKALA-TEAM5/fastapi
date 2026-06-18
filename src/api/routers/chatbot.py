# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-04
# 수정일   : 2026-06-18
#
# [ 주요 엔드포인트 정의 ]
#
# 1. POST /chat : 산안비 챗봇 질의응답 (SSE 스트리밍)
#
# [ 응답 형식 ]
# Content-Type: text/event-stream
# 이벤트 순서:
#   data: {"type": "session_id", "value": "..."}
#   data: {"type": "intent", "value": "카테고리판단"}
#   data: {"type": "token", "value": "안전모는"}
#   data: {"type": "token", "value": " CAT_03"}
#   ...
#   data: {"type": "sources", "value": ["산업안전보건법 제72조"]}
#   data: [DONE]
# --------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.chatbot import ChatRequest
from src.services.chatbot_service import stream_chat

router = APIRouter(prefix="/chat", tags=["챗봇"])


@router.post(
    "",
    summary="산안비 챗봇 질의응답",
    description="""
산업안전보건관리비(산안비) 관련 질문에 대해 RAG 기반 답변을 SSE 스트리밍으로 반환합니다.

**지원 질문 유형**
- 카테고리 판단: "안전모는 몇 번 카테고리인가요?"
- 법령 한도: "CAT_02 안전시설비 한도율이 얼마인가요?"
- 적법성 판단: "냉방기 구입이 산안비로 인정되나요?"

**session_id**
- 전달하면 해당 세션의 이전 대화를 이어받습니다.
- 미전달 시 서버에서 UUID를 생성하여 `session_id` 이벤트로 반환합니다.
- 탭 닫기 또는 서버 재시작 시 대화 기록이 소멸합니다.
    """,
    response_description="SSE 스트리밍 응답 (text/event-stream)",
)
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream chatbot responses through the stable SSE contract."""
    return StreamingResponse(
        stream_chat(question=req.question, session_id=req.session_id, user_id=req.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )
