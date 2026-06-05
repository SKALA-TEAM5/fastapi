# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-06-02
#
# FastAPI 애플리케이션 진입점
# - Spring Backend와의 통신을 위한 CORS 설정
# - 포트 8001 (k8s: team5-fastapi:8001)
# --------------------------------------------------------------------------
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import (
    matching,
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

app = FastAPI(title="AI Workspace", version="0.1.0")

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


app.include_router(orchestrator.router)
app.include_router(validation.router)
app.include_router(report_agent.router)
app.include_router(parse.router)
app.include_router(matching.router)
app.include_router(receipts.router)
app.include_router(tax_invoices.router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
