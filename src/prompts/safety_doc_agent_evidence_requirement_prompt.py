from __future__ import annotations

import json

from fastapi.src.schemas.safety_doc_agent_evidence import AIEvidenceRequirementInput


SYSTEM_PROMPT = """
당신은 산업안전보건관리비 증빙서류 판단 AI입니다.
주어진 품목과 증빙유형 정의를 보고, 실제로 필요한 증빙유형 코드만 고르세요.
반드시 available_evidence_types 안에 있는 코드만 반환하세요.
이름(name)이나 설명(description)이 아니라 code만 반환하세요.
과도하게 많이 고르지 말고, 최소 필요 증빙 중심으로 판단하세요.
반드시 JSON으로만 응답하세요.
""".strip()


def build_user_prompt(payload: AIEvidenceRequirementInput) -> str:
    """DB에서 가져온 항목과 증빙 정의로 독립 실행 가능한 프롬프트를 만든다."""

    body = {
        "item_context": {
            "project_id": payload.item_context.project_id,
            "project_name": payload.item_context.project_name,
            "usage_statement_id": payload.item_context.usage_statement_id,
            "report_month": payload.item_context.report_month,
            "revision_no": payload.item_context.revision_no,
            "item_id": payload.item_context.item_id,
            "category_code": payload.item_context.category_code,
            "category_name": payload.item_context.category_name,
            "item_name": payload.item_context.item_name,
            "used_on": payload.item_context.used_on,
            "unit": payload.item_context.unit,
            "quantity": payload.item_context.quantity,
            "unit_price": payload.item_context.unit_price,
            "total_amount": payload.item_context.total_amount,
            "remark": payload.item_context.remark,
            "page_no": payload.item_context.page_no,
        },
        "linked_files": [
            {
                "file_id": linked_file.file_id,
                "original_filename": linked_file.original_filename,
                "mime_type": linked_file.mime_type,
                "uploaded_evidence_type_code": linked_file.uploaded_evidence_type_code,
                "linked_evidence_type_code": linked_file.linked_evidence_type_code,
                "storage_key": linked_file.storage_key,
                "captured_at": linked_file.captured_at,
                "uploaded_at": linked_file.uploaded_at,
            }
            for linked_file in payload.linked_files
        ],
        "available_evidence_types": payload.available_evidence_types,
        "evidence_type_definitions": [
            {
                "code": evidence_type.code,
                "name": evidence_type.name,
                "description": evidence_type.description,
            }
            for evidence_type in payload.evidence_type_definitions
        ],
        "output_schema": {
            "required_evidences": ["evidence_type_code"],
            "confidence": 0.0,
            "reason": "왜 이 증빙이 필요한지 짧은 설명",
        },
    }
    return json.dumps(body, ensure_ascii=False, indent=2)
