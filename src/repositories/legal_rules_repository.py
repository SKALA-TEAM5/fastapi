# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. LegalRulesRepository : 법령 payload / rulebook 조회 저장소
# 2. find_category_candidates() : classifier용 카테고리 후보 검색
# 3. find_validator_matches() : validator용 규칙 매칭
# 4. find_category_limit() : 카테고리 한도 및 근거 조회
# --------------------------------------------------------------------------
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from src.schemas.classifier import CATEGORIES

load_dotenv()

DEFAULT_RULES_PATH = Path("artifacts/legal_rules_payload.json")
DEFAULT_RULE_CONFIG_PATH = Path("config/legal_rule_profiles.json")
DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://safety_user:safety_password@localhost:5432/safety",
)


def _load_static_rule_config(rule_config_path: Path = DEFAULT_RULE_CONFIG_PATH) -> dict:
    if not rule_config_path.exists():
        return {
            "validator_synonyms": {},
            "validator_profiles": {},
            "classifier_profiles": {},
            "generic_item_policies": {},
        }
    return json.loads(rule_config_path.read_text(encoding="utf-8"))


_STATIC_RULE_CONFIG = _load_static_rule_config()
_ACTIVE_VALIDATOR_SYNONYMS = {
    key: {str(value).lower() for value in values}
    for key, values in _STATIC_RULE_CONFIG.get("validator_synonyms", {}).items()
}


@dataclass
class CategoryCandidate:
    category_code: str
    category_name: str
    score: float
    evidence: list[str]


@dataclass
class ValidatorRuleMatch:
    category_code: str
    category_name: str
    rule_type: str
    allowed: bool | None
    score: float
    evidence: str
    referenced_laws: list[str]
    limit_pct: float | None = None
    source_id: str = ""


class LegalRulesRepository:
    def __init__(
        self,
        payload_path: str | Path = DEFAULT_RULES_PATH,
        rule_config_path: str | Path = DEFAULT_RULE_CONFIG_PATH,
        database_url: str = DEFAULT_DATABASE_URL,
    ) -> None:
        self.payload_path = Path(payload_path)
        self.rule_config_path = Path(rule_config_path)
        self.database_url = database_url
        self._payload: dict | None = None
        self._rule_config: dict | None = None
        self._rule_index: list[dict] | None = None
        self._token_df: dict[str, int] | None = None

    def _load(self) -> dict:
        if self._payload is not None:
            return self._payload
        conn = psycopg2.connect(self.database_url)
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        master_id, source_id, rule_type, category_code, category_name,
                        item_key AS keyword, item_pattern, body AS rule_text,
                        legal_basis, allowed, limit_pct, cited_laws, keywords, metadata
                    FROM legal_rag.legal_rule_master
                    WHERE record_type = 'rule'
                """)
                rows = cur.fetchall()
        conn.close()
        rules = []
        for row in rows:
            r = dict(row)
            r["cited_laws"] = list(r.get("cited_laws") or [])
            r["keywords"] = list(r.get("keywords") or [])
            metadata = dict(r.get("metadata") or {})
            r["metadata"] = metadata
            # V2 매핑 이전 원본 rule_type 복원 (스코어링 로직 유지)
            original_rule_type = metadata.get("original_rule_type")
            if original_rule_type:
                r["rule_type"] = original_rule_type
            if r.get("limit_pct") is not None:
                r["limit_pct"] = float(r["limit_pct"])
            rules.append(r)
        self._payload = {"rules": rules}
        return self._payload

    def _load_rule_config(self) -> dict:
        if self._rule_config is not None:
            return self._rule_config
        conn = psycopg2.connect(self.database_url)
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT profile_scope, category_code, profile_key, values_json, metadata
                    FROM legal_rag.legal_rule_profiles
                """)
                rows = cur.fetchall()
        conn.close()
        config: dict = {
            "validator_synonyms": {},
            "validator_profiles": {},
            "classifier_profiles": {},
            "generic_item_policies": {},
        }
        for row in rows:
            key = row["profile_key"]
            values = row["values_json"]
            category = row["category_code"]
            metadata = dict(row.get("metadata") or {})
            # V2 매핑 이전 원본 scope 복원 (global→validator_synonym, category→validator/classifier_profile, item→generic)
            scope = metadata.get("original_scope", row["profile_scope"])
            if scope == "validator_synonym":
                config["validator_synonyms"][key] = values
            elif scope == "validator_profile" and category:
                config["validator_profiles"].setdefault(category, {})[key] = values
            elif scope == "classifier_profile" and category:
                config["classifier_profiles"].setdefault(category, {})[key] = values
            elif scope == "generic_item_policy":
                config["generic_item_policies"][key] = values
        self._rule_config = config
        return self._rule_config

    def _build_index(self) -> None:
        if self._rule_index is not None and self._token_df is not None:
            return

        rule_index: list[dict] = []
        token_df: dict[str, int] = {}
        for rule in self.rules:
            category_code = rule.get("category_code")
            if not category_code or category_code not in CATEGORIES:
                continue
            if rule.get("rule_type") not in _CLASSIFIER_RULE_TYPES:
                continue

            text = " ".join(
                str(rule.get(key, "") or "")
                for key in ("keyword", "item_pattern", "rule_text", "legal_basis", "category_name")
            )
            tokens = _tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                token_df[token] = token_df.get(token, 0) + 1

            rule_index.append(
                {
                    "category_code": category_code,
                    "rule_type": rule.get("rule_type", ""),
                    "tokens": tokens,
                    "evidence": _clean_evidence(rule.get("rule_text") or rule.get("keyword") or ""),
                }
            )

        self._rule_index = rule_index
        self._token_df = token_df

    @property
    def rules(self) -> list[dict]:
        return self._load().get("rules", [])

    @property
    def validator_synonyms(self) -> dict[str, set[str]]:
        raw = self._load_rule_config().get("validator_synonyms", {})
        return {
            key: {str(value).lower() for value in values}
            for key, values in raw.items()
        }

    @property
    def validator_profiles(self) -> dict[str, dict[str, list[str]]]:
        return self._load_rule_config().get("validator_profiles", {})

    @property
    def classifier_profiles(self) -> dict[str, dict]:
        return self._load_rule_config().get("classifier_profiles", {})

    @property
    def generic_item_policies(self) -> dict[str, dict]:
        return self._load_rule_config().get("generic_item_policies", {})

    def find_category_candidates(
        self,
        *,
        query_text: str,
        retrieved_context: str = "",
        limit: int = 5,
    ) -> list[CategoryCandidate]:
        self._build_index()

        query_text_norm = _normalize_text(query_text)
        query_tokens = _tokenize(query_text)
        context_tokens = _tokenize(retrieved_context)
        categories: dict[str, CategoryCandidate] = {
            code: CategoryCandidate(
                category_code=code,
                category_name=name,
                score=0.0,
                evidence=[],
            )
            for code, name in CATEGORIES.items()
        }

        for category_code, profile in self.classifier_profiles.items():
            score, evidence = _score_profile(
                category_code=category_code,
                query_text=query_text_norm,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                profile=profile,
            )
            if score <= 0:
                continue
            candidate = categories[category_code]
            candidate.score += score
            for item in evidence:
                if item not in candidate.evidence:
                    candidate.evidence.append(item)

        for rule in self._rule_index or []:
            score, evidence = _score_rule(
                rule,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                token_df=self._token_df or {},
                corpus_size=len(self._rule_index or []),
            )
            if score <= 0:
                continue

            category_code = rule["category_code"]
            candidate = categories[category_code]
            candidate.score += score
            if evidence and evidence not in candidate.evidence:
                candidate.evidence.append(evidence)

        ranked = [candidate for candidate in categories.values() if candidate.score > 0]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def find_category_hints(
        self,
        *,
        category_codes: list[str],
        limit_per_category: int = 6,
    ) -> dict[str, dict[str, list[str]]]:
        hints: dict[str, dict[str, list[str]]] = {}
        wanted = {code for code in category_codes if code in CATEGORIES}
        if not wanted:
            return hints

        for category_code in wanted:
            cited_laws: list[str] = []
            keywords: list[str] = []
            seen_laws: set[str] = set()
            seen_keywords: set[str] = set()

            for rule in self.rules:
                if rule.get("category_code") != category_code:
                    continue
                for law in _rule_laws(rule, category_code):
                    law_norm = str(law).strip()
                    if law_norm and law_norm not in seen_laws:
                        seen_laws.add(law_norm)
                        cited_laws.append(law_norm)

                candidates = list(rule.get("keywords") or [])
                keyword = str(rule.get("keyword") or "").strip()
                pattern = str(rule.get("item_pattern") or "").strip()
                if keyword:
                    candidates.append(keyword)
                if pattern:
                    candidates.append(pattern)

                for token in candidates:
                    token_norm = str(token).strip()
                    if (
                        token_norm
                        and len(token_norm) >= 2
                        and token_norm not in seen_keywords
                    ):
                        seen_keywords.add(token_norm)
                        keywords.append(token_norm)

                if len(cited_laws) >= limit_per_category and len(keywords) >= limit_per_category:
                    break

            hints[category_code] = {
                "cited_laws": cited_laws[:limit_per_category],
                "keywords": keywords[:limit_per_category],
            }

        return hints

    def resolve_category(self, category: str) -> tuple[str | None, str]:
        if category in CATEGORIES:
            return category, CATEGORIES[category]
        for code, name in CATEGORIES.items():
            if name == category:
                return code, name
        return None, category

    def find_category_limit(self, category: str) -> tuple[float | None, str, list[str]]:
        category_code, category_name = self.resolve_category(category)
        candidates: list[tuple[float, float | None, str, list[str]]] = []
        for rule in self.rules:
            if category_code and rule.get("category_code") != category_code:
                continue
            if not _is_rule_in_category(rule, category_code=category_code, category_name=category_name):
                continue
            limit_pct = rule.get("limit_pct")
            metadata = rule.get("metadata") or {}
            limit_text = metadata.get("limit_rule_text") or rule.get("rule_text") or ""
            if limit_pct is None and not any(keyword in limit_text for keyword in _LIMIT_HINTS):
                continue
            score = _validator_rule_type_weight(rule.get("rule_type", ""))
            candidates.append(
                (
                    score,
                    limit_pct,
                    _clean_evidence(limit_text),
                    _rule_laws(rule, category_code),
                )
            )

        if not candidates:
            return None, "", _primary_laws(category_code)

        candidates.sort(key=lambda item: (item[1] is not None, item[0]), reverse=True)
        _, limit_pct, text, laws = candidates[0]
        return limit_pct, text, laws

    def find_validator_matches(
        self,
        *,
        category: str,
        item_text: str,
        retrieved_context: str = "",
        limit: int = 8,
    ) -> list[ValidatorRuleMatch]:
        category_code, category_name = self.resolve_category(category)
        item_text_norm = _normalize_text(item_text)
        query_tokens = _validator_tokens(item_text)
        context_tokens = _validator_tokens(retrieved_context)
        matches: list[ValidatorRuleMatch] = []

        profile = self.validator_profiles.get(category_code or "", {})
        disallow_hits = _profile_hits(item_text_norm, profile.get("disallow_terms", []))
        allow_hits = _profile_hits(item_text_norm, profile.get("allow_terms", []))

        for rule in self.rules:
            if not _is_rule_in_category(rule, category_code=category_code, category_name=category_name):
                continue
            normalized_allowed = _normalized_validator_allowed(
                rule=rule,
                item_text=item_text_norm,
                disallow_hits=disallow_hits,
            )
            score = _score_validator_rule(
                rule=rule,
                category_code=category_code,
                item_tokens=query_tokens,
                context_tokens=context_tokens,
                item_text=item_text_norm,
                disallow_hits=disallow_hits,
                allow_hits=allow_hits,
                normalized_allowed=normalized_allowed,
            )
            if score <= 0:
                continue

            matches.append(
                ValidatorRuleMatch(
                    category_code=category_code or rule.get("category_code") or "",
                    category_name=category_name,
                    rule_type=rule.get("rule_type", ""),
                    allowed=normalized_allowed,
                    score=score,
                    evidence=_clean_evidence(rule.get("rule_text") or rule.get("keyword") or ""),
                    referenced_laws=_rule_laws(rule, category_code),
                    limit_pct=rule.get("limit_pct"),
                    source_id=rule.get("source_id", ""),
                )
            )

        has_disallowed_match = any(match.allowed is False for match in matches)
        has_allowed_match = any(match.allowed is True for match in matches)
        if disallow_hits and not has_disallowed_match:
            fallback_match = _build_fallback_validator_match(
                category_code=category_code,
                category_name=category_name,
                item_text=item_text_norm,
                allow_hits=set(),
                disallow_hits=disallow_hits,
                validator_profiles=self.validator_profiles,
            )
            if fallback_match is not None:
                matches.append(fallback_match)
        elif allow_hits and not has_allowed_match and not matches:
            fallback_match = _build_fallback_validator_match(
                category_code=category_code,
                category_name=category_name,
                item_text=item_text_norm,
                allow_hits=allow_hits,
                disallow_hits=set(),
                validator_profiles=self.validator_profiles,
            )
            if fallback_match is not None:
                matches.append(fallback_match)

        matches.sort(key=lambda item: item.score, reverse=True)
        if not matches:
            fallback_match = _build_fallback_validator_match(
                category_code=category_code,
                category_name=category_name,
                item_text=item_text_norm,
                allow_hits=allow_hits,
                disallow_hits=disallow_hits,
                validator_profiles=self.validator_profiles,
            )
            if fallback_match is not None:
                matches.append(fallback_match)
        return matches[:limit]


_CLASSIFIER_RULE_TYPES = {"category", "allowed", "qa_allowed", "limit", "qa_limit"}
_VALIDATOR_RULE_TYPES = {
    "category",
    "allowed",
    "disallowed",
    "qa_allowed",
    "qa_disallowed",
    "limit",
    "qa_limit",
    "rule_like_allowed",
    "rule_like_disallowed",
}
_LIMIT_HINTS = ("초과", "%", "분의", "이내")
_PRIMARY_LAW_BY_CATEGORY = {
    "CAT_01": "제7조제1항제1호",
    "CAT_02": "제7조제1항제2호",
    "CAT_03": "제7조제1항제3호",
    "CAT_04": "제7조제1항제4호",
    "CAT_05": "제7조제1항제5호",
    "CAT_06": "제7조제1항제6호",
    "CAT_07": "제7조제1항제7호",
    "CAT_08": "제7조제1항제8호",
    "CAT_09": "제7조제1항제9호",
}
_STOPWORDS = {
    "건설", "건설공사", "건설현장", "현장", "근로자", "산업", "안전", "보건", "산업안전", "산업안전보건",
    "관리비", "비용", "구입", "임대", "설치", "사용", "가능", "가능한지", "항목", "소요", "해당", "따른",
    "법", "규정", "기준", "제", "호", "목", "등", "위한", "위하여", "대한", "경우", "업무", "관련", "실시",
    "관리", "예방", "필요", "장비", "시설", "현수", "세트", "인건비",
    "등에", "따라", "위해", "대해", "로서", "로써", "에서", "에게", "에도", "으로", "부터", "까지",
}

def _tokenize(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text or "")
        if len(token) >= 2 and token.lower() not in _STOPWORDS
    }
    return tokens


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^0-9a-z가-힣\s]+", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _clean_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:160]


def _score_profile(
    *,
    category_code: str,
    query_text: str,
    query_tokens: set[str],
    context_tokens: set[str],
    profile: dict,
) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []

    for left, right in profile.get("pair_terms", []):
        if left.lower() in query_text and right.lower() in query_text:
            score += 8.0
            evidence.append(f"질의에 '{left}+{right}' 조합이 직접 포함됨")

    for term in profile.get("strong_terms", set()):
        if term.lower() in query_text:
            score += 5.0
            evidence.append(f"질의에 강한 분류 신호 '{term}' 포함")

    for term in profile.get("medium_terms", set()):
        if term.lower() in query_text:
            score += 2.5
            evidence.append(f"질의에 보조 분류 신호 '{term}' 포함")

    negative_hits = {term for term in profile.get("negative_terms", set()) if term.lower() in query_text}
    if negative_hits:
        score -= 2.5 * len(negative_hits)
        evidence.append(f"비선호 신호 {', '.join(sorted(negative_hits))} 포함")

    if category_code == "CAT_03" and "안전인증" in query_text and "스티커" not in query_text:
        score += 2.0
    if category_code == "CAT_06" and "kf94" in query_text:
        score += 6.0
        evidence.append("질의에 'KF94'가 포함되어 건강장해예방비 신호가 강함")
    if category_code == "CAT_04" and any(token.endswith("측정기") for token in query_tokens):
        score += 6.0
        evidence.append("품목명이 '측정기' 계열이라 진단/측정비 신호가 강함")
    if category_code == "CAT_02" and "설치" in query_text and "인건비" in query_text:
        score += 4.0
        evidence.append("시설 설치 인건비 패턴과 일치")
    if category_code == "CAT_05" and "스티커" in query_tokens:
        score += 2.0
        evidence.append("직접 규정은 약하지만 표식/배포물 성격으로 교육비 후보")
    if category_code == "CAT_06" and {"코로나19", "진단키트"} & query_tokens:
        score += 5.0
        evidence.append("감염병/진단키트 문맥이 건강장해예방비와 직접 연결됨")

    # 검색 문맥은 보조 근거로만 아주 약하게 반영한다.
    context_overlap = context_tokens & query_tokens
    if context_overlap:
        score += min(len(context_overlap), 3) * 0.3

    return score, evidence[:5]


def _score_rule(
    rule: dict,
    *,
    query_tokens: set[str],
    context_tokens: set[str],
    token_df: dict[str, int],
    corpus_size: int,
) -> tuple[float, str]:
    rule_tokens = rule["tokens"]
    overlap = query_tokens & rule_tokens
    if not overlap:
        return 0.0, ""

    weight_by_type = {
        "category": 1.8,
        "allowed": 1.4,
        "qa_allowed": 1.1,
        "limit": 0.8,
        "qa_limit": 0.7,
    }
    base = weight_by_type.get(rule["rule_type"], 0.5)

    score = 0.0
    for token in overlap:
        df = token_df.get(token, 1)
        idf = 1.0 + math.log((1 + corpus_size) / (1 + df))
        length_bonus = min(len(token) / 4.0, 2.0)
        score += base * idf * length_bonus

    # 검색 문맥은 규칙이 이미 맞물린 뒤에만 약한 추가 점수로 사용한다.
    context_overlap = (context_tokens & rule_tokens) - overlap
    if context_overlap:
        score += 0.15 * min(len(context_overlap), 3)

    return score, rule["evidence"]


def _validator_tokens(text: str) -> set[str]:
    base = _tokenize(text)
    expanded = set(base)
    for token in list(base):
        for key, values in _ACTIVE_VALIDATOR_SYNONYMS.items():
            if token == key.lower() or token in values:
                expanded |= values
    return expanded


def _profile_hits(item_text: str, terms: list[str]) -> set[str]:
    return {term for term in terms if term.lower() in item_text}


def _validator_rule_type_weight(rule_type: str) -> float:
    return {
        "disallowed": 3.0,
        "qa_disallowed": 2.8,
        "allowed": 2.2,
        "category": 2.0,
        "qa_allowed": 1.9,
        "limit": 1.6,
        "qa_limit": 1.5,
        "rule_like_allowed": 1.2,
        "rule_like_disallowed": 1.2,
    }.get(rule_type, 0.8)


def _is_rule_in_category(rule: dict, *, category_code: str | None, category_name: str) -> bool:
    if rule.get("rule_type") not in _VALIDATOR_RULE_TYPES:
        return False
    if category_code and rule.get("category_code") == category_code:
        return True
    text = " ".join(
        str(rule.get(key, "") or "")
        for key in ("category_name", "keyword", "item_pattern", "rule_text")
    )
    if category_name and category_name in text:
        return True
    category_keywords = _validator_tokens(category_name)
    rule_tokens = _validator_tokens(text)
    return bool(category_keywords & rule_tokens)


def _primary_laws(category_code: str | None) -> list[str]:
    law = _PRIMARY_LAW_BY_CATEGORY.get(category_code or "")
    return [law] if law else []


def _rule_laws(rule: dict, category_code: str | None) -> list[str]:
    laws = _primary_laws(category_code)
    legal_basis = str(rule.get("legal_basis") or "").strip()
    if legal_basis and legal_basis not in laws:
        laws.append(legal_basis)
    rule_text = str(rule.get("rule_text") or "")
    for law in re.findall(r"제\d+조(?:제\d+항)?(?:제\d+호)?", rule_text):
        if law not in laws:
            laws.append(law)
    return laws


def _score_validator_rule(
    *,
    rule: dict,
    category_code: str | None,
    item_tokens: set[str],
    context_tokens: set[str],
    item_text: str,
    disallow_hits: set[str],
    allow_hits: set[str],
    normalized_allowed: bool | None,
) -> float:
    rule_text = " ".join(
        str(rule.get(key, "") or "")
        for key in ("keyword", "item_pattern", "rule_text", "category_name", "legal_basis")
    )
    rule_tokens = _validator_tokens(rule_text)
    if not rule_tokens:
        return 0.0

    overlap = item_tokens & rule_tokens
    if not overlap and not any(hit.lower() in rule_text.lower() for hit in disallow_hits | allow_hits):
        return 0.0

    base = _validator_rule_type_weight(rule.get("rule_type", ""))
    score = base * len(overlap)

    for token in overlap:
        score += min(len(token) / 3.0, 2.5)

    text_lower = rule_text.lower()
    if normalized_allowed is False:
        score += 2.0 * sum(1 for hit in disallow_hits if hit.lower() in text_lower)
        if not disallow_hits:
            score *= 0.35
    if normalized_allowed is True:
        score += 1.5 * sum(1 for hit in allow_hits if hit.lower() in text_lower)
        if disallow_hits and any(hit.lower() in text_lower for hit in disallow_hits) and "불가" in text_lower:
            score *= 0.55

    if category_code == "CAT_03" and disallow_hits and normalized_allowed is False:
        if "근로자 재해" in text_lower or "원활한 공사수행" in text_lower:
            score += 4.0
    if category_code == "CAT_05" and normalized_allowed is False and not disallow_hits:
        score *= 0.2
    if category_code == "CAT_09" and normalized_allowed is False and not disallow_hits:
        score *= 0.2

    context_overlap = (context_tokens & rule_tokens) - overlap
    if context_overlap:
        score += min(len(context_overlap), 2) * 0.25

    if "다만" in rule_text or "불가" in rule_text:
        score += 0.5
    if "가능" in rule_text or "허용" in rule_text:
        score += 0.3
    return score


def _normalized_validator_allowed(
    *,
    rule: dict,
    item_text: str,
    disallow_hits: set[str],
) -> bool | None:
    text = _normalize_text(
        " ".join(
            str(rule.get(key, "") or "")
            for key in ("keyword", "item_pattern", "rule_text")
        )
    )
    allowed = rule.get("allowed")

    if "불가" in text:
        if disallow_hits and any(hit.lower() in text for hit in disallow_hits):
            return False
        if "근로자 재해" in text and any(term in item_text for term in ("감리원", "감리자", "방문자")):
            return False

    if rule.get("rule_type") in {"disallowed", "qa_disallowed", "rule_like_disallowed"}:
        return False

    return allowed


def _build_fallback_validator_match(
    *,
    category_code: str | None,
    category_name: str,
    item_text: str,
    allow_hits: set[str],
    disallow_hits: set[str],
    validator_profiles: dict[str, dict[str, list[str]]],
) -> ValidatorRuleMatch | None:
    if not category_code:
        return None
    profile = validator_profiles.get(category_code, {})
    if disallow_hits:
        return ValidatorRuleMatch(
            category_code=category_code,
            category_name=category_name,
            rule_type="profile_disallowed",
            allowed=False,
            score=4.5 + len(disallow_hits),
            evidence=(
                f"{category_name} 카테고리에서는 '{', '.join(sorted(disallow_hits))}' 관련 항목이 "
                "예외 또는 제한 조건으로 다뤄질 수 있습니다."
            ),
            referenced_laws=_primary_laws(category_code),
        )
    if allow_hits:
        return ValidatorRuleMatch(
            category_code=category_code,
            category_name=category_name,
            rule_type="profile_allowed",
            allowed=True,
            score=4.0 + len(allow_hits),
            evidence=(
                f"{category_name} 카테고리의 일반 허용 범위와 "
                f"'{', '.join(sorted(allow_hits))}' 신호가 일치합니다."
            ),
            referenced_laws=_primary_laws(category_code),
        )
    profile_terms = {term.lower() for term in profile.get("allow_terms", [])}
    for term in profile_terms:
        if term and term in item_text:
            return ValidatorRuleMatch(
                category_code=category_code,
                category_name=category_name,
                rule_type="profile_allowed",
                allowed=True,
                score=4.0,
                evidence=f"{category_name} 카테고리의 일반 허용 범위와 '{term}' 신호가 일치합니다.",
                referenced_laws=_primary_laws(category_code),
            )
    return None
