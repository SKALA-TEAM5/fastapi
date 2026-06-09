# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-02
#
# FastAPI 애플리케이션 진입점
# - 모든 라우터를 /api/v1 prefix로 등록
# - Spring Backend와의 통신을 위한 CORS 설정
# - 포트 8001 (k8s: team5-fastapi:8001)
# --------------------------------------------------------------------------
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import (
    chatbot,
    link,
    orchestrator,
    parse,
    receipts,
    report_agent,
    tax_invoices,
    validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 HuggingFace 모델과 LangGraph 그래프를 미리 로딩한다.

    classifier / validator / chatbot 세 에이전트가 공통으로 사용하는
    jhgan/ko-sroberta-multitask (임베딩)과 BAAI/bge-reranker-v2-m3 (reranker)를
    싱글톤 캐시에 올려두어 첫 요청 지연을 없앤다.
    """
    log.info("워밍업 시작 — HuggingFace 모델 로딩 중...")
    try:
        from src.agents.chatbot_agent.agent import get_compiled_graph
        from src.core.rag import _get_rerank_model
        from src.core.storage import DEFAULT_COLLECTION, load_vectorstore

        load_vectorstore(DEFAULT_COLLECTION)  # jhgan/ko-sroberta-multitask
        _get_rerank_model()  # BAAI/bge-reranker-v2-m3
        get_compiled_graph()  # LangGraph 챗봇 그래프
        log.info("워밍업 완료 — 모든 모델이 메모리에 로딩되었습니다.")
    except Exception as e:
        log.warning(f"워밍업 실패 (서버는 정상 기동): {e}")

    yield  # 서버가 요청을 받기 시작


app = FastAPI(title="AI Workspace", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://team5-iveri.skala25a.project.skala-ai.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Spring Backend의 현재 FastAPI client는 /api/v1 없이 /orchestrator 경로를 호출한다.
# 외부 문서/테스트에서 쓰는 /api/v1/orchestrator 경로도 유지하기 위해 두 prefix를 함께 연다.
app.include_router(orchestrator.router)
app.include_router(orchestrator.router, prefix="/api/v1")
app.include_router(validation.router, prefix="/api/v1")
app.include_router(report_agent.router, prefix="/api/v1")
app.include_router(parse.router, prefix="/api/v1")
app.include_router(link.router, prefix="/api/v1")
app.include_router(receipts.router, prefix="/api/v1")
app.include_router(tax_invoices.router, prefix="/api/v1")
app.include_router(chatbot.router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
