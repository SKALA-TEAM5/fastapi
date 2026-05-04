from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.validator_agent.context_retriever import CategoryRetrievedContext
from src.agents.validator_agent.parser import CategoryInputBlock, CategoryItemRow
from src.repositories import LegalRulesRepository, ValidatorRuleMatch

_PROGRESS_RULE_LAW = "별표 3 공사진척에 따른 산업안전보건관리비 사용기준"


@dataclass
class ItemRuleBundle:
    item: CategoryItemRow
    matches: list[ValidatorRuleMatch]
    context_text: str
    item_exception_text: str = ""

    @property
    def top_allowed(self) -> ValidatorRuleMatch | None:
        for match in self.matches:
            if match.allowed is True:
                return match
        return None

    @property
    def top_disallowed(self) -> ValidatorRuleMatch | None:
        for match in self.matches:
            if match.allowed is False:
                return match
        return None

    @property
    def has_exception(self) -> bool:
        return any(keyword in self.item_exception_text for keyword in ("단,", "다만", "제외", "불가"))


@dataclass
class CategoryRuleBundle:
    category_code: str
    category_name: str
    limit_pct: float | None
    limit_rule: str
    primary_laws: list[str] = field(default_factory=list)
    progress_law: str = _PROGRESS_RULE_LAW
    items: list[ItemRuleBundle] = field(default_factory=list)


def match_category_rules(
    *,
    block: CategoryInputBlock,
    retrieved: CategoryRetrievedContext,
    repo: LegalRulesRepository | None = None,
) -> CategoryRuleBundle:
    rules_repo = repo or LegalRulesRepository()
    limit_pct, limit_rule, limit_laws = rules_repo.find_category_limit(block.category_name)

    item_bundles: list[ItemRuleBundle] = []
    for item in block.items:
        docs = retrieved.item_docs.get(item.item_name) or []
        context_parts = [doc.page_content for doc in (docs or retrieved.category_docs)]
        context_text = "\n\n---\n\n".join(context_parts)
        item_exception_text = "\n\n---\n\n".join(
            doc.page_content for doc in docs
            if any(keyword in (doc.page_content or "") for keyword in ("단,", "다만", "제외", "불가"))
        )
        matches = rules_repo.find_validator_matches(
            category=block.category_name,
            item_text=item.item_name,
            retrieved_context=context_text,
            limit=8,
        )
        item_bundles.append(
            ItemRuleBundle(
                item=item,
                matches=matches,
                context_text=context_text,
                item_exception_text=item_exception_text,
            )
        )

    return CategoryRuleBundle(
        category_code=block.category_code,
        category_name=block.category_name,
        limit_pct=limit_pct,
        limit_rule=limit_rule,
        primary_laws=limit_laws,
        items=item_bundles,
    )
