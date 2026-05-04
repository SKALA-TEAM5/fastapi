from langchain_core.language_models.chat_models import BaseChatModel

_llm: BaseChatModel | None = None


def configure(llm: BaseChatModel) -> None:
    """사용할 LLM을 설정합니다. 노드 실행 전 반드시 호출해야 합니다."""
    global _llm
    _llm = llm


def get() -> BaseChatModel:
    if _llm is None:
        raise RuntimeError(
            "LLM이 설정되지 않았습니다. llm_config.configure(llm)를 먼저 호출하세요."
        )
    return _llm
