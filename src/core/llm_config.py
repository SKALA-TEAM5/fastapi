# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. configure() : 전역 LLM 인스턴스 설정
# 2. get()       : 현재 설정된 전역 LLM 인스턴스 반환
# --------------------------------------------------------------------------
from __future__ import annotations

import os
from threading import Lock

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

_llm: BaseChatModel | None = None
_lock = Lock()
_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

load_dotenv()


def configure(llm: BaseChatModel) -> None:
    """사용할 LLM을 설정합니다. 노드 실행 전 반드시 호출해야 합니다."""
    global _llm
    _llm = llm


def get() -> BaseChatModel:
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "LLM이 설정되지 않았습니다. OPENAI_API_KEY 또는 llm_config.configure(llm)가 필요합니다."
                    )
                _llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)
    return _llm
