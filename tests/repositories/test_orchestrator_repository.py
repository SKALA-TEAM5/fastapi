from contextlib import contextmanager

from src.repositories import orchestrator_repository
from src.repositories.orchestrator_repository import SITE_PHOTO_TYPES


def test_vision_evidence_types_include_supported_photo_categories():
    assert {
        "site_photo",
        "item_photo",
        "wearing_photo",
        "work_photo",
        "tech_guidance_photo",
    }.issubset(SITE_PHOTO_TYPES)


def test_upsert_agent_log_does_not_reuse_item_level_log(monkeypatch):
    executed_sql: list[str] = []

    class Cursor:
        def execute(self, sql, params):
            executed_sql.append(sql)

        def fetchone(self):
            return None if len(executed_sql) == 1 else (101,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Connection:
        def cursor(self):
            return Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    @contextmanager
    def fake_connection():
        yield Connection()

    monkeypatch.setattr(orchestrator_repository, "get_connection", fake_connection)

    log_id = orchestrator_repository.upsert_agent_log(
        project_id=8,
        usage_statement_id=9,
        agent_type_code="safety-doc",
        status_code="success",
    )

    assert log_id == 101
    assert "usage_statement_item_id IS NULL" in executed_sql[0]
