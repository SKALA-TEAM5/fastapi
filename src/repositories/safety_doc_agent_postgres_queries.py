"""`EvidenceRepository` 구현 시 참고할 PostgreSQL SQL 모음.

`SKALA-TEAM5/db`의 `service` 스키마를 기준으로 작성되었고,
FastAPI 서비스 레이어에서 실제 DB 어댑터를 붙일 때 시작점으로 쓸 수 있다.
"""

GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT = """
SELECT
    project_id,
    project_name,
    item_id,
    usage_statement_id,
    report_month::text AS report_month,
    revision_no,
    category_code,
    category_name,
    used_on::text AS used_on,
    item_name,
    unit,
    quantity::float8 AS quantity,
    unit_price::float8 AS unit_price,
    total_amount::bigint AS total_amount,
    remark,
    page_no
FROM service.v_ai_evidence_requirement_item_context
WHERE item_id = %(item_id)s
"""

GET_EVIDENCE_REQUIREMENT_ITEM_CONTEXT_FALLBACK = """
SELECT
    p.id AS project_id,
    p.project_name,
    usi.id AS item_id,
    usi.usage_statement_id,
    us.report_month::text AS report_month,
    us.revision_no,
    usi.category_code,
    uc.name AS category_name,
    usi.used_on::text AS used_on,
    usi.item_name,
    usi.unit,
    usi.quantity::float8 AS quantity,
    usi.unit_price::float8 AS unit_price,
    usi.total_amount::bigint AS total_amount,
    usi.remark,
    usi.page_no
FROM service.usage_statement_items usi
JOIN service.usage_statements us
  ON us.id = usi.usage_statement_id
JOIN service.projects p
  ON p.id = us.project_id
JOIN service.usage_categories uc
  ON uc.code = usi.category_code
WHERE usi.id = %(item_id)s
"""

LIST_LINKED_FILE_CONTEXTS = """
SELECT
    item_id,
    file_id,
    original_filename,
    mime_type,
    uploaded_evidence_type_code,
    linked_evidence_type_code,
    storage_key,
    captured_at::text AS captured_at,
    uploaded_at::text AS uploaded_at
FROM service.v_ai_evidence_requirement_file_context
WHERE item_id = %(item_id)s
ORDER BY uploaded_at NULLS LAST, file_id
"""

LIST_LINKED_FILE_CONTEXTS_FALLBACK = """
SELECT
    efl.usage_statement_item_id AS item_id,
    f.id AS file_id,
    f.original_filename,
    f.mime_type,
    f.uploaded_evidence_type_code,
    efl.evidence_type_code AS linked_evidence_type_code,
    f.storage_key,
    f.captured_at::text AS captured_at,
    f.uploaded_at::text AS uploaded_at
FROM service.evidence_file_links efl
JOIN service.files f
  ON f.id = efl.file_id
WHERE efl.usage_statement_item_id = %(item_id)s
ORDER BY f.uploaded_at NULLS LAST, f.id
"""

LIST_EVIDENCE_TYPES = """
SELECT
    code,
    name,
    description
FROM service.evidence_types
ORDER BY code
"""

DEACTIVATE_ACTIVE_REQUIREMENTS = """
UPDATE service.evidence_requirements
SET
    is_active = false,
    updated_at = now()
WHERE usage_statement_item_id = %(item_id)s
  AND is_active = true
"""

INSERT_REQUIREMENT = """
INSERT INTO service.evidence_requirements
    (usage_statement_item_id, evidence_type_code, is_satisfied, is_active)
VALUES
    (%(item_id)s, %(evidence_type_code)s, false, true)
RETURNING id, usage_statement_item_id, evidence_type_code, is_satisfied, is_active
"""

LIST_ACTIVE_REQUIREMENTS = """
SELECT
    id,
    usage_statement_item_id,
    evidence_type_code,
    is_satisfied,
    is_active
FROM service.evidence_requirements
WHERE usage_statement_item_id = %(item_id)s
  AND is_active = true
ORDER BY evidence_type_code
"""

LIST_EVIDENCE_LINKS = """
SELECT
    usage_statement_item_id,
    file_id,
    evidence_type_code
FROM service.evidence_file_links
WHERE usage_statement_item_id = %(item_id)s
ORDER BY created_at
"""

MARK_SATISFIED_REQUIREMENTS = """
UPDATE service.evidence_requirements
SET
    is_satisfied = CASE
        WHEN evidence_type_code = ANY(%(submitted_codes)s) THEN true
        ELSE false
    END,
    updated_at = now()
WHERE usage_statement_item_id = %(item_id)s
  AND is_active = true
"""

INSERT_AGENT_LOG = """
INSERT INTO service.agent_logs
    (
        project_id,
        usage_statement_id,
        usage_statement_item_id,
        agent_type_code,
        status_code,
        result_code,
        reason,
        details,
        model_name,
        token
    )
VALUES
    (
        %(project_id)s,
        %(usage_statement_id)s,
        %(usage_statement_item_id)s,
        'safety-doc',
        %(status_code)s,
        %(result_code)s,
        %(reason)s,
        %(details)s::jsonb,
        %(model_name)s,
        %(token)s
    )
"""

LIST_ITEM_CONTEXT_TARGETS_FALLBACK = """
SELECT
    p.id AS project_id,
    p.project_name,
    us.id AS usage_statement_id,
    us.report_month::text AS report_month,
    us.revision_no,
    usi.id AS item_id,
    usi.category_code,
    uc.name AS category_name,
    usi.item_name,
    usi.used_on::text AS used_on,
    usi.total_amount::bigint AS total_amount
FROM service.usage_statement_items usi
JOIN service.usage_statements us
  ON us.id = usi.usage_statement_id
JOIN service.projects p
  ON p.id = us.project_id
JOIN service.usage_categories uc
  ON uc.code = usi.category_code
ORDER BY p.id, us.report_month DESC, us.revision_no DESC, usi.id
LIMIT %(limit)s
"""
