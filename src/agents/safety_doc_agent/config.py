from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """DB 기반 safety-doc-agent 실행 설정."""

    openai_api_key: str
    chat_model: str = "gpt-4.1-mini"
    langsmith_tracing: bool = False
    langsmith_project: str = ""
    langsmith_workspace_id: str = ""
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "safety"
    db_user: str = "safety_service_app"
    db_password: str = "safety_service_app_password"


def app_root_dir() -> Path:
    """에이전트 앱의 루트 디렉토리를 반환한다.

    현재 패키지 구조는 `src/agents/safety_doc_agent/`이므로, 이 파일 기준으로
    세 단계 위가 앱 루트다. 실행 위치가 달라도 데이터 파일과 설정 탐색 기준을
    고정하려고 별도 함수로 분리한다.
    """

    return Path(__file__).resolve().parents[3]


def _load_nearest_env() -> None:
    """현재 작업 경로와 앱 상위 경로를 따라가며 가장 가까운 `.env`를 읽는다.

    `skala-final-project/ai`처럼 하위 앱으로 옮겨도 상위 워크스페이스의 `.env`를
    그대로 재사용할 수 있게 하려는 목적이다.
    """

    candidates: list[Path] = []

    for base in (Path.cwd(), app_root_dir(), app_root_dir().parent):
        candidates.extend(path / ".env" for path in (base, *base.parents))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return

    load_dotenv()


def load_settings() -> Settings:
    """실행 시작 시 환경변수 기반 설정을 한 번에 읽는다.

    초기에 실패시키면 이후 로직은 필수 키와 저장소 설정이 이미
    준비되어 있다고 가정할 수 있다.
    """

    _load_nearest_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    return Settings(
        openai_api_key=api_key,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip(),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true",
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "").strip(),
        langsmith_workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID", "").strip(),
        db_host=os.getenv("POSTGRES_HOST", "localhost").strip(),
        db_port=int(os.getenv("POSTGRES_PORT", "5432").strip()),
        db_name=os.getenv("POSTGRES_DB", "safety").strip(),
        db_user=os.getenv("SERVICE_APP_USER", "safety_service_app").strip(),
        db_password=os.getenv("SERVICE_APP_PASSWORD", "safety_service_app_password").strip(),
    )
