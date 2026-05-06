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
