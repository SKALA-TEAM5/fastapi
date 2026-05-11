from __future__ import annotations

from src.repositories.safety_doc_agent_evidence_repository import EvidenceRepository
from src.schemas.safety_doc_agent_evidence import EvidenceStatusDetail, EvidenceStatusResult


class EvidenceCheckService:
    """필수 증빙과 제출된 파일 연결 정보를 비교한다."""

    def __init__(self, repository: EvidenceRepository) -> None:
        self.repository = repository

    def run(self, item_id: int) -> EvidenceStatusResult:
        """상세 항목 1건의 증빙 만족 상태를 다시 계산한다."""

        requirements = self.repository.list_active_requirements(item_id)
        links = self.repository.list_evidence_links(item_id)
        submitted_codes = sorted({link.evidence_type_code for link in links})
        self.repository.update_requirement_satisfaction(item_id, submitted_codes)

        requirement_codes = sorted({requirement.evidence_type_code for requirement in requirements})
        missing_codes = [code for code in requirement_codes if code not in submitted_codes]

        details = [
            EvidenceStatusDetail(
                evidence_type_code=code,
                required=True,
                submitted=code in submitted_codes,
            )
            for code in requirement_codes
        ]

        return EvidenceStatusResult(
            item_id=item_id,
            status="MISSING" if missing_codes else "OK",
            missing_evidences=missing_codes,
            submitted_evidences=submitted_codes,
            details=details,
        )
