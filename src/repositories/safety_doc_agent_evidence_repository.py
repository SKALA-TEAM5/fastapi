from __future__ import annotations

from typing import Protocol

from src.schemas.safety_doc_agent_evidence import (
    EvidenceRequirementItemContext,
    EvidenceFileLink,
    EvidenceRequirement,
    EvidenceType,
    LinkedEvidenceFileContext,
)


class EvidenceRepository(Protocol):
    """AI 계층이 사용하는 `service` 스키마 접근 규약.

    실제 DB 드라이버와 분리해두면 FastAPI 쪽에서 psycopg나 SQLAlchemy
    구현체를 붙여도 서비스 로직은 이 계약만 의존하면 된다.
    """

    def get_item_context(self, item_id: int) -> EvidenceRequirementItemContext:
        """AI 판단에 필요한 항목 문맥 뷰 1건을 조회한다."""

    def list_evidence_types(self) -> list[EvidenceType]:
        """선택 가능한 전체 증빙 유형 코드를 조회한다."""

    def list_linked_file_contexts(self, item_id: int) -> list[LinkedEvidenceFileContext]:
        """AI 판단에 참고할 연결 파일 문맥 뷰 목록을 조회한다."""

    def replace_active_requirements(
        self,
        item_id: int,
        evidence_type_codes: list[str],
    ) -> list[EvidenceRequirement]:
        """이전 active requirement를 비활성화하고 새 active 집합을 저장한다."""

    def list_active_requirements(self, item_id: int) -> list[EvidenceRequirement]:
        """항목 1건에 대한 현재 active requirement 집합을 조회한다."""

    def list_evidence_links(self, item_id: int) -> list[EvidenceFileLink]:
        """항목 1건에 연결된 제출 증빙 목록을 조회한다."""

    def update_requirement_satisfaction(
        self,
        item_id: int,
        satisfied_codes: list[str],
    ) -> None:
        """제출된 증빙 코드 기준으로 active requirement 만족 여부를 갱신한다."""

    def append_agent_log(
        self,
        *,
        project_id: int,
        usage_statement_id: int | None,
        usage_statement_item_id: int | None,
        status_code: str,
        result_code: str,
        reason: str,
        details: dict,
        model_name: str | None,
        token: int | None,
    ) -> None:
        """`service.agent_logs`에 safety-doc agent 실행 로그를 저장한다."""
