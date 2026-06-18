# --------------------------------------------------------------------------
# 작성자   : 이현수(kacalu0930)
# 작성일   : 2026-05-11
# 수정일   : 2026-06-18 (Clova→VLM 전환: OCR_ENGINE / CLOVA_OCR_URL / CLOVA_OCR_SECRET 제거)
#
# [ 모듈 설명 ]
#   환경변수 기반 전역 설정 상수 모듈 (함수 없음).
#
# [ 주요 설정 ]
#   - VLM_PROVIDER / GEMINI_API_KEY / OPENAI_API_KEY : OCR용 VLM 프로바이더·API 키
#   - (그 외 DB · MinIO · Qdrant 등 인프라 환경설정)
# --------------------------------------------------------------------------
"""
산업안전관리비 AI 검증 시스템 — 전역 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
.env 파일 또는 환경변수에서 설정값을 읽어온다.
모든 모듈은 이 파일을 통해 설정값에 접근한다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드 (없으면 상위 디렉토리까지 탐색 — skala 모노레포 구조 대응)
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")  # fastapi/.env 우선
load_dotenv(
    _ROOT.parent / ".env"
)  # skala/.env fallback (override=False 기본값 — 이미 로드된 값은 덮어쓰지 않음)


# ══════════════════════════════════════════════
# VLM — Gemini / OpenAI (OCR 엔진)
# ══════════════════════════════════════════════
# [Clova→VLM 리팩토링] 기존 OCR_ENGINE / CLOVA_OCR_URL / CLOVA_OCR_SECRET 설정 제거.
#   OCR은 VLM 전면 사용으로 전환되어 Clova 관련 환경변수는 더 이상 필요 없음.

VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "gemini").lower()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_MODEL_FALLBACK: str = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-3.5-flash")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_FALLBACK: str = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")


# ══════════════════════════════════════════════
# 매칭 임계값
# ══════════════════════════════════════════════

THRESHOLD_MATCHED: float = float(os.getenv("THRESHOLD_MATCHED", "0.85"))
THRESHOLD_REVIEW: float = float(os.getenv("THRESHOLD_REVIEW", "0.75"))


# ══════════════════════════════════════════════
# PostgreSQL (service 스키마)
# ══════════════════════════════════════════════

DB_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME: str = os.getenv("POSTGRES_DB", "safety")
DB_USER: str = os.getenv("SERVICE_APP_USER", "safety_user")
DB_PASSWORD: str = os.getenv("SERVICE_APP_PASSWORD", "safety_password")
DB_SCHEMA: str = "service"
LAW_DB_USER: str = os.getenv("LAW_APP_USER", "safety_law_app")
LAW_DB_PASSWORD: str = os.getenv("LAW_APP_PASSWORD", "safety_law_password")
LAW_DB_SCHEMA: str = "legal_rag"

DATABASE_URL: str = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?options=-csearch_path%3D{DB_SCHEMA}"
)

LEGAL_DATABASE_URL: str = (
    f"postgresql://{LAW_DB_USER}:{LAW_DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?options=-csearch_path%3D{LAW_DB_SCHEMA}"
)


# ══════════════════════════════════════════════
# Qdrant 벡터 DB
# ══════════════════════════════════════════════

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # Cloud 사용 시에만 필요

# ※ 컬렉션 이름은 여기서 관리하지 않습니다.
#   QdrantRepository(collection_name="...") 형태로 사용 시점에 직접 전달하세요.


# ══════════════════════════════════════════════
# AWS S3 / MinIO
# ══════════════════════════════════════════════

AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv(
    "APP_MINIO_ACCESS_KEY", ""
)
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv(
    "APP_MINIO_SECRET_KEY", ""
)
AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET: str = os.getenv("S3_BUCKET") or os.getenv(
    "APP_MINIO_BUCKET", ""
)  # 필수 — 버킷 이름 확정 후 .env에 설정
S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL") or os.getenv(
    "APP_MINIO_ENDPOINT", ""
)
S3_PUBLIC_ENDPOINT_URL: str = (
    os.getenv("S3_PUBLIC_ENDPOINT_URL")
    or os.getenv("APP_MINIO_PUBLIC_ENDPOINT")
    or S3_ENDPOINT_URL
)
S3_PRESIGNED_URL_EXPIRE_SECONDS: int = int(
    os.getenv("S3_PRESIGNED_URL_EXPIRE_SECONDS", "900")
)


# ══════════════════════════════════════════════
# External Agents
# ══════════════════════════════════════════════

VISION_AGENT_BASE_URL: str = os.getenv("VISION_AGENT_BASE_URL", "")
VISION_AGENT_REVIEW_PATH: str = os.getenv("VISION_AGENT_REVIEW_PATH", "/vision/review")
VISION_AGENT_TIMEOUT_SECONDS: int = int(os.getenv("VISION_AGENT_TIMEOUT_SECONDS", "60"))
