import json
from types import SimpleNamespace

import pytest

from src.services import chatbot_service


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _payload(raw: str) -> dict:
    assert raw.startswith("data: ")
    assert raw != "data: [DONE]\n\n"
    return json.loads(raw.removeprefix("data: "))


class _FakeGraph:
    def __init__(self, events: list[dict], messages: list | None = None):
        self.events = events
        self.messages = messages or []
        self.initial_state = None
        self.config = None

    async def astream_events(self, initial_state, *, config, version):
        self.initial_state = initial_state
        self.config = config
        for event in self.events:
            yield event

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self.messages})


@pytest.mark.anyio
async def test_stream_chat_preserves_sse_sequence_and_usage(monkeypatch):
    saved_messages: list[list] = []
    token_records: list[dict] = []
    finished: list[dict] = []

    events = [
        {"event": "on_chain_start", "name": "intent_classifier", "data": {}},
        {"event": "on_chain_end", "name": "intent_classifier", "data": {"output": {"intent": "카테고리판단"}}},
        {"event": "on_chain_start", "name": "answer_generator", "data": {}},
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "answer_generator"},
            "data": {"chunk": SimpleNamespace(content="안전모는")},
        },
        {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "metadata": {"langgraph_node": "answer_generator"},
            "data": {"chunk": SimpleNamespace(content=" CAT_03입니다.")},
        },
        {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "data": {
                "output": SimpleNamespace(
                    usage_metadata={"input_tokens": 5, "output_tokens": 7},
                    response_metadata={"model_name": "chat-test"},
                )
            },
        },
        {
            "event": "on_chain_end",
            "name": "answer_generator",
            "data": {"output": {"sources": ["산업안전보건법 제72조"]}},
        },
    ]
    graph = _FakeGraph(events, messages=["persisted"])

    monkeypatch.setattr(chatbot_service, "get_compiled_graph", lambda: graph)
    monkeypatch.setattr(chatbot_service, "_has_session_state", lambda session_id: False)
    monkeypatch.setattr(chatbot_service, "_load_chatbot_messages", lambda session_id: [])
    monkeypatch.setattr(chatbot_service, "_save_chatbot_messages", lambda session_id, messages: saved_messages.append(messages))
    monkeypatch.setattr(chatbot_service, "start_agent_run", lambda agent: "started")
    monkeypatch.setattr(chatbot_service, "finish_agent_run", lambda **kwargs: finished.append(kwargs))
    monkeypatch.setattr(chatbot_service, "record_agent_tokens", lambda **kwargs: token_records.append(kwargs))

    raw_events = [
        raw
        async for raw in chatbot_service.stream_chat(
            question="안전모는 몇 번인가요?",
            session_id="session-1",
        )
    ]

    assert raw_events[-1] == "data: [DONE]\n\n"
    payloads = [_payload(raw) for raw in raw_events[:-1]]
    assert [row["type"] for row in payloads] == [
        "session_id",
        "session_reset",
        "status",
        "intent",
        "status",
        "token",
        "token",
        "sources",
    ]
    assert payloads[0]["value"] == "session-1"
    assert payloads[3]["value"] == "카테고리판단"
    assert payloads[5]["value"] == "안전모는"
    assert payloads[6]["value"] == " CAT_03입니다."
    assert payloads[7]["value"] == ["산업안전보건법 제72조"]
    assert graph.initial_state["messages"] == []
    assert saved_messages == [["persisted"]]
    assert {row["token_type"]: row["value"] for row in token_records} == {
        "input": 5,
        "output": 7,
        "total": 12,
    }
    assert all(row["model"] == "chat-test" for row in token_records)
    assert finished == [{"agent": "chatbot", "started_at": "started", "result": "success"}]


@pytest.mark.anyio
async def test_stream_chat_duplicate_request_returns_error_and_done(monkeypatch):
    session_id = "locked-session"
    lock = chatbot_service._get_session_lock(session_id)
    records: list[dict] = []

    await lock.acquire()
    try:
        monkeypatch.setattr(chatbot_service, "record_agent_run", lambda **kwargs: records.append(kwargs))
        raw_events = [
            raw
            async for raw in chatbot_service.stream_chat(
                question="중복 요청",
                session_id=session_id,
            )
        ]
    finally:
        lock.release()
        chatbot_service._session_locks.pop(session_id, None)

    payload = _payload(raw_events[0])
    assert payload["type"] == "error"
    assert "이전 답변" in payload["value"]
    assert raw_events[1] == "data: [DONE]\n\n"
    assert records == [{"agent": "chatbot", "result": "skipped"}]
