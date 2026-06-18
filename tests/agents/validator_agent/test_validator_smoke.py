from src.agents.validator_agent import agent as validator_agent
from src.agents.validator_agent import audit, rule_matcher
from src.agents.validator_agent.parser import parse_usage_statement, resolve_category
from src.agents.validator_agent.presenter import summarize_audit_response
from src.repositories import ValidatorRuleMatch
from src.schemas.validator import AuditResponse, CategoryAuditResult, ItemJudgment


def _sample_document() -> dict:
    return {
        "사용내역서ID": 3,
        "기본정보": {
            "산안비총액": 1000000,
            "누계공정률": 50,
        },
        "카테고리별데이터": [
            {
                "카테고리코드": "CAT_02",
                "집계정보": {
                    "전회사용금액": 0,
                    "금회사용금액": 100000,
                    "누적사용금액": 100000,
                },
                "항목목록": [
                    {
                        "행ID": 11,
                        "항목명": "안전난간 설치",
                        "금액": 100000,
                        "비고": "추락 예방",
                    }
                ],
            }
        ],
    }


def _audit_response() -> AuditResponse:
    item = ItemJudgment(
        item="안전난간 설치",
        amount=100000,
        category="안전시설비 등",
        allowed=True,
        confidence=0.91,
        reasoning="제7조제1항제2호에 따른 안전시설 설치비로 확인됩니다.",
        referenced_laws=["제7조제1항제2호"],
        reason_text="안전난간 설치는 산업재해 예방을 위한 안전시설 설치비로 집행 가능합니다.",
    )
    result = CategoryAuditResult(
        status="적절",
        total=100000,
        limit=None,
        exceeded=False,
        items=[item],
        referenced_laws=["제7조제1항제2호"],
    )
    return AuditResponse(
        base_amount=1000000,
        categories={"안전시설비 등": result},
        total_token_usage=0,
    )


def test_parse_usage_statement_normalizes_category_block():
    parsed = parse_usage_statement(_sample_document())

    assert parsed.usage_statement_id == 3
    assert parsed.base_amount == 1000000
    assert parsed.progress_rate == 50
    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.category_code == "CAT_02"
    assert block.category_name == "안전시설비 등"
    assert block.items[0].row_id == 11
    assert block.items[0].item_name == "안전난간 설치"
    assert block.items[0].amount == 100000


def test_resolve_category_accepts_code_and_display_name():
    assert resolve_category("CAT_02") == ("CAT_02", "안전시설비 등")
    assert resolve_category("안전시설비 등") == ("CAT_02", "안전시설비 등")
    assert resolve_category("기타") == (None, "기타")


def test_validate_usage_statement_keeps_public_wrapper_contract(monkeypatch):
    expected = _audit_response()

    def fake_validate_usage_statement_service(*, document, collection):
        assert document["사용내역서ID"] == 3
        assert collection == "legal_documents"
        return expected

    monkeypatch.setattr(
        validator_agent,
        "validate_usage_statement_service",
        fake_validate_usage_statement_service,
    )

    actual = validator_agent.validate_usage_statement(
        document=_sample_document(),
        collection="legal_documents",
    )

    assert actual is expected


def test_summarize_audit_response_preserves_orchestrator_shape():
    summary = summarize_audit_response(
        response=_audit_response(),
        usage_statement_id=3,
    )

    assert summary.usage_statement_id == 3
    assert len(summary.results) == 1
    result = summary.results[0]
    assert result.category_code == "CAT_02"
    assert result.status == "적절"
    assert "안전난간 설치" in result.reason
    assert result.sources
    assert "제7조" in result.sources[0].law


def test_process_single_item_uses_strong_rdb_allowed_path(monkeypatch):
    match = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="allowed",
        allowed=True,
        score=4.0,
        evidence="안전난간 설치는 추락 예방을 위한 안전시설 설치비입니다.",
        referenced_laws=["제7조제1항제2호"],
        source_id="rule-1",
        match_source="law_rule",
    )

    class FakeRepo:
        def find_validator_matches(self, **kwargs):
            assert kwargs["item_text"] == "안전난간 설치"
            return [match]

    class FakeDoc:
        page_content = "[LEGAL_CITE:x] 제7조제1항제2호 안전시설 설치비"
        metadata = {"source": "legal-source"}

    monkeypatch.setattr(
        rule_matcher,
        "_llm_generate_reason_only",
        lambda **kwargs: "RDB 근거 기반 사유",
    )

    parsed = parse_usage_statement(_sample_document())
    block = parsed.blocks[0]
    retrieved = rule_matcher.CategoryRetrievedContext(
        category_docs=[FakeDoc()],
        exception_docs=[],
        item_docs={"안전난간 설치": [FakeDoc()]},
    )

    bundle = rule_matcher._process_single_item(
        item=block.items[0],
        block=block,
        retrieved=retrieved,
        rules_repo=FakeRepo(),
    )

    assert bundle.judgment_tier == "rdb"
    assert bundle.matches[0] is match
    assert bundle.reason_text == "RDB 근거 기반 사유"
    assert bundle.qdrant_citations[0]["judgment_source"] == "qdrant_support"
    assert "[LEGAL_CITE" not in bundle.context_text


def test_process_single_item_uses_llm_fallback_when_rdb_is_weak(monkeypatch):
    weak_match = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="weak",
        allowed=True,
        score=0.5,
        evidence="약한 토큰 매칭",
        referenced_laws=[],
        source_id="weak-1",
        match_source="law_rule",
    )
    llm_match = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="llm_judgment",
        allowed=False,
        score=6.2,
        evidence="",
        referenced_laws=["제7조제1항제2호"],
        source_id="llm_fallback",
        match_source="llm_fallback",
    )

    class FakeRepo:
        def find_validator_matches(self, **kwargs):
            return [weak_match]

    class FakeDoc:
        page_content = "사무실 비치용 물품은 사용 불가"
        metadata = {"source": "legal-source"}

    monkeypatch.setattr(
        rule_matcher,
        "_llm_item_fallback",
        lambda **kwargs: (llm_match, "LLM fallback 사유"),
    )

    parsed = parse_usage_statement(_sample_document())
    block = parsed.blocks[0]
    retrieved = rule_matcher.CategoryRetrievedContext(
        category_docs=[FakeDoc()],
        exception_docs=[],
        item_docs={},
    )

    bundle = rule_matcher._process_single_item(
        item=block.items[0],
        block=block,
        retrieved=retrieved,
        rules_repo=FakeRepo(),
    )

    assert bundle.judgment_tier == "llm"
    assert bundle.matches[0] is llm_match
    assert bundle.matches[1] is weak_match
    assert bundle.reason_text == "LLM fallback 사유"
    assert bundle.qdrant_citations[0]["judgment_source"] == "llm_fallback"


def test_build_item_judgment_prefers_disallowed_when_score_wins():
    allowed = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="allowed",
        allowed=True,
        score=2.0,
        evidence="일반 안전시설은 허용됩니다.",
        referenced_laws=["제7조제1항제2호"],
        source_id="allow-1",
        match_source="law_rule",
    )
    disallowed = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="disallowed",
        allowed=False,
        score=4.0,
        evidence="사무실 비치용 소화기는 사용 불가합니다.",
        referenced_laws=["별표 2"],
        source_id="deny-1",
        match_source="law_rule",
    )
    item = parse_usage_statement(_sample_document()).blocks[0].items[0]
    bundle = rule_matcher.ItemRuleBundle(
        item=item,
        matches=[disallowed, allowed],
        context_text="",
    )

    judgment = audit._build_item_judgment(bundle, category_name="안전시설비 등")

    assert judgment.allowed is False
    assert judgment.referenced_laws == ["별표 2"]
    assert judgment.source_ids == ["deny-1"]
    assert judgment.force_reason_text is True


def test_build_item_judgment_keeps_signage_allowed_even_with_generic_disallow():
    disallowed = ValidatorRuleMatch(
        category_code="CAT_02",
        category_name="안전시설비 등",
        rule_type="disallowed",
        allowed=False,
        score=5.0,
        evidence="안전시설물 설치비용은 사용 불가합니다.",
        referenced_laws=["별표 2"],
        source_id="deny-1",
        match_source="law_rule",
    )
    item = parse_usage_statement(_sample_document()).blocks[0].items[0]
    item.item_name = "안전표지판 설치"
    bundle = rule_matcher.ItemRuleBundle(
        item=item,
        matches=[disallowed],
        context_text="",
    )

    judgment = audit._build_item_judgment(bundle, category_name="안전시설비 등")

    assert judgment.allowed is True
    assert judgment.needs_human_review is False
    assert judgment.force_reason_text is False
