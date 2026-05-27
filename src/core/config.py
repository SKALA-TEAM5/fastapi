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
load_dotenv(_ROOT / ".env")          # fastapi/.env 우선
load_dotenv(_ROOT.parent / ".env")   # skala/.env fallback (override=False 기본값 — 이미 로드된 값은 덮어쓰지 않음)


# ══════════════════════════════════════════════
# CLOVA OCR
# ══════════════════════════════════════════════

CLOVA_OCR_URL: str = os.getenv("CLOVA_OCR_URL", "")
CLOVA_OCR_SECRET: str = os.getenv("CLOVA_OCR_SECRET", "")


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

DATABASE_URL: str = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?options=-csearch_path%3D{DB_SCHEMA}"
)


# ══════════════════════════════════════════════
# Qdrant 벡터 DB
# ══════════════════════════════════════════════

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # Cloud 사용 시에만 필요

# ※ 컬렉션 이름은 여기서 관리하지 않습니다.
#   QdrantRepository(collection_name="...") 형태로 사용 시점에 직접 전달하세요.


# ══════════════════════════════════════════════
# AWS S3
# ══════════════════════════════════════════════

AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET: str = os.getenv("S3_BUCKET", "")  # 필수 — 버킷 이름 확정 후 .env에 설정
