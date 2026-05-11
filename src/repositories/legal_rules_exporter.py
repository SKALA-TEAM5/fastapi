# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. build_payload() : 법령/해설/프로필 통합 payload 생성
# 2. payload_to_sql() : payload를 PostgreSQL seed SQL로 변환
# 3. main() : export 및 선택적 DB 적용 진입점
# --------------------------------------------------------------------------
import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

# TODO(refresh/postgres): PostgreSQL이 주 저장소가 되면 이 모듈은
# 초기 적재/백필 스크립트로 축소하거나, refresh 파이프라인의 정규화 단계로 재사용한다.

_CITE_PATTERN = re.compile(r"\[LEGAL_CITE:\s*([^\]]+)\]")
_INLINE_CITATION_RE = re.compile(
    r"(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?(?:\s*[가-하]\s*목)?|별표\s*\d+(?:의\s*\d+)?|별지\s*제?\s*\d+\s*호\s*서식)"
)
_ARTICLE_HEADER_RE = re.compile(r"^제(\d+)조(?:\(([^)]+)\))?")
_APPENDIX_ROW_RE = re.compile(
    r"^\[LEGAL_CITE:\s*([^\]]+)\]\s*\|\s*(\d+)\.\s*([^|]+?)\s*\|\s*(.+?)\|?$"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_EFFECTIVE_DATE_RE = re.compile(r"\[시행\s+(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\]")
_NOTICE_NO_RE = re.compile(r"\[고용노동부고시\s+제([^\]]+)\]")
_PCT_PATTERNS = [
    (re.compile(r"100분의\s*(\d+)"), lambda m: int(m.group(1)) / 100),
    (re.compile(r"20분의\s*(\d+)"), lambda m: int(m.group(1)) / 20),
    (re.compile(r"10분의\s*(\d+)"), lambda m: int(m.group(1)) / 10),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), lambda m: float(m.group(1)) / 100),
]
_LIMIT_KEYWORDS = [
    "초과 불가",
    "초과할 수 없",
    "초과할수없",
    "이내",
    "를 넘을 수 없",
    "초과 금지",
    "초과금지",
]
_TOTAL_KEYWORDS = ["총액", "계상액"]
_NOISE_PREFIXES = ("법제처", "국가법령정보센터", "Ministry of", "Ministry of Employment")
_NOISE_LINES = {"③ <삭 제>", "③ &lt;삭 제&gt;"}
_NOISE_RE = re.compile(
    r"^\d+$"                                      # 페이지 번호
    r"|^\d{4}\.\s*\d{1,2}$"                       # 연도.월 (예: 2025. 6)
    r"|^\d{3}-\d{3,4}-\d{4}$"                     # 전화번호
    r"|^\[시행\s+\d{4}.*?\].*$"                    # [시행 날짜] 단독 줄
    r"|^\[고용노동부고시\s+제.*?\].*$"              # [고용노동부고시...] 단독 줄
)
_COMMENTARY_ARTICLE_RE = re.compile(r"【고시\s+(제\d+조)】|【법\s+(제\d+조)】")

# V2 Flyway 스키마 CHECK constraint 매핑
_V2_SOURCE_TYPE: dict[str, str | None] = {
    "law_notice": "law",
    "appendix_disallowed": "law",
    "commentary": "guideline",
    "rule_config": None,
}
_V2_CONTENT_TYPE: dict[str, str] = {
    "section": "article",
    "commentary": "guideline",
}
_V2_RULE_TYPE: dict[str, str] = {
    "rule_like": "progress",
    "rule_like_allowed": "allowed",
    "rule_like_disallowed": "disallowed",
    "rule_like_limit": "limit",
    "category": "progress",
    "qa_allowed": "qa",
    "qa_disallowed": "qa",
    "qa_limit": "qa",
}
_V2_PROFILE_SCOPE: dict[str, str] = {
    "validator_synonym": "global",
    "validator_profile": "category",
    "classifier_profile": "category",
    "generic_item_policy": "item",
}

_CATEGORY_CODES = {
    1: "CAT_01",
    2: "CAT_02",
    3: "CAT_03",
    4: "CAT_04",
    5: "CAT_05",
    6: "CAT_06",
    7: "CAT_07",
    8: "CAT_08",
    9: "CAT_09",
}

_CATEGORY_NAMES = {
    1: "안전관리자 등의 인건비 및 각종 업무 수당 등",
    2: "안전시설비 등",
    3: "보호구 등",
    4: "안전보건진단비 등",
    5: "안전보건교육비 등",
    6: "근로자 건강장해예방비 등",
    7: "건설재해예방 기술지도비",
    8: "본사 안전전담부서 운영비",
    9: "위험성평가 등에 따른 소요비용",
}

_QUESTION_PATTERN = re.compile(r"^(?:####\s+|-?\s*)?(\d+)\)\s+(.+)$")
_REGULATORY_TOKENS = [
    "하여야 한다",
    "따른다",
    "사용이 가능",
    "사용 가능",
    "사용이 불가",
    "사용 불가",
    "할 수 있다",
    "할 수 없다",
    "초과할 수 없다",
    "이내",
    "준수하여야",
    "지급하는 비용",
    "소요되는 비용",
    "구입비용",
    "임대 비용",
    "설치비용",
    "사용기준",
]

_RULE_TEXT_KEYWORDS = ("해당", "사용", "불가", "가능", "초과", "이내", "지급", "구입", "임대", "설치", "비용")
_QUESTIONISH_SUFFIXES = (
    "사용 가능한지",
    "사용이 가능한지",
    "사용 불가한지",
    "가능한지",
    "불가한지",
    "되는지",
    "있는지",
)
_PROGRESS_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)퍼센트\s*이상(?:\s*(\d+(?:\.\d+)?)퍼센트\s*미만)?")
_PROGRESS_USAGE_RE = re.compile(r"(\d+(?:\.\d+)?)퍼센트\s*이상")
_APPENDIX_1_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)%|(\d{1,3}(?:,\d{3})+)원")
_APPENDIX_LIST_ITEM_RE = re.compile(r"^(\d+)\.\s*(.+)$")


@dataclass
class SourceDocument:
    source_id: str
    source_name: str
    source_type: str
    source_path: str
    title: str | None
    effective_date: str | None
    notice_no: str | None


@dataclass
class CorpusEntry:
    corpus_id: str
    source_id: str
    content_type: str
    title: str | None
    article_no: str | None
    section_path: str | None
    body: str
    cited_laws: list[str]
    metadata: dict


@dataclass
class LegalRule:
    rule_id: str
    source_id: str
    rule_type: str
    category_code: str | None
    category_number: int | None
    category_name: str | None
    allowed: bool | None
    keyword: str | None
    item_pattern: str | None
    legal_basis: str | None
    limit_pct: float | None
    rule_text: str
    metadata: dict


@dataclass
class LegalCitation:
    citation_id: str
    source_id: str
    parent_type: str
    parent_id: str
    sequence_no: int
    citation_text: str
    article_no: str | None
    paragraph_no: str | None
    item_no: str | None
    subitem_no: str | None


@dataclass
class LegalRuleProfile:
    profile_id: str
    profile_scope: str
    category_code: str | None
    profile_key: str
    values_json: dict | list
    metadata: dict


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if slug:
        return slug
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_rule_text_for_storage(text: str, *, rule_type: str | None = None) -> str:
    cleaned = _normalize_whitespace(_CITE_PATTERN.sub("", text or ""))
    if not cleaned:
        return ""

    cleaned = cleaned.replace("으로 사용 가능한지 으로 사용 가능한지", "으로 사용 가능한지")
    cleaned = cleaned.replace("비 용", "비용")
    cleaned = re.sub(r"^귀\s+질의의\s*", "", cleaned)
    cleaned = re.sub(
        r"^(?:다만,\s*)?(?:귀\s*)?질의(?:의|내용)?만으로[^,.]*정확한 답변을 드리기 어려우나,\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:질의\s*내용만으로|귀\s*질의만으로)[^,.]*정확한 답변을 드리기 어려우나,\s*",
        "",
        cleaned,
    )

    segments = _split_rule_text_segments(cleaned)
    if not segments:
        return cleaned

    preferred = max(segments, key=_rule_text_segment_score)
    preferred = preferred.strip(" -")

    if "사용 가능한지" in preferred:
        tail = preferred.split("사용 가능한지", 1)[1].strip(" :-")
        if tail:
            preferred = tail
    if preferred.startswith("가능한지"):
        preferred = preferred[len("가능한지"):].strip(" :-")

    preferred = _normalize_whitespace(preferred)
    if preferred and not preferred.endswith((".", "다", "함")) and len(preferred) < 160:
        preferred = preferred.rstrip(" ,")
    return preferred or cleaned


def _parse_progress_range_cell(text: str) -> tuple[float, float | None] | None:
    normalized = _normalize_whitespace(text)
    match = _PROGRESS_RANGE_RE.search(normalized)
    if not match:
        return None
    min_rate = float(match.group(1))
    max_rate = float(match.group(2)) if match.group(2) else None
    return min_rate, max_rate


def _parse_progress_usage_cell(text: str) -> float | None:
    normalized = _normalize_whitespace(text)
    match = _PROGRESS_USAGE_RE.search(normalized)
    if not match:
        return None
    return float(match.group(1)) / 100


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [_normalize_whitespace(cell) for cell in stripped.split("|")]


def _is_question_like_rule_text(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return True
    if normalized.endswith("?"):
        return True
    if any(normalized.endswith(suffix) for suffix in _QUESTIONISH_SUFFIXES):
        return True
    return bool(re.search(r"(?:인지|한지|되는지|하는지|여부)$", normalized))


def _split_rule_text_segments(text: str) -> list[str]:
    raw_segments = re.split(r"\s*(?:-\s+|\s+|•\s+|\u25cf\s+|\u25a3\s+)\s*", text)
    segments: list[str] = []
    for raw in raw_segments:
        seg = _normalize_whitespace(raw)
        seg = re.sub(r"^[0-9]+[.)]?\s*", "", seg)
        seg = seg.strip(" -")
        if seg:
            segments.append(seg)
    return segments


def _rule_text_segment_score(text: str) -> tuple[int, int, int]:
    score = 0
    if "사용 가능한지" in text:
        score -= 4
    if "질의" in text or "문의" in text:
        score -= 2
    score += sum(1 for keyword in _RULE_TEXT_KEYWORDS if keyword in text)
    if any(token in text for token in ("불가", "초과", "이내", "해당", "가능")):
        score += 2
    return (score, min(len(text), 160), -text.count("「"))


def _split_cites(raw: str | None) -> list[str]:
    if not raw:
        return []
    cites: list[str] = []
    for part in raw.split("|"):
        cite = _normalize_whitespace(part)
        if cite and cite not in cites:
            cites.append(cite)
    return cites


def _extract_cites(text: str) -> list[str]:
    cites: list[str] = []
    for raw in _CITE_PATTERN.findall(text):
        for cite in _split_cites(raw):
            if cite not in cites:
                cites.append(cite)
    for match in _INLINE_CITATION_RE.findall(text):
        cite = _normalize_whitespace(match)
        if cite and cite not in cites:
            cites.append(cite)
    return cites


def _parse_citation_parts(citation: str) -> tuple[str | None, str | None, str | None, str | None]:
    normalized = _normalize_whitespace(citation)

    article_match = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", normalized)
    paragraph_match = re.search(r"제\s*(\d+)\s*항", normalized)
    item_match = re.search(r"제\s*(\d+)\s*호", normalized)
    subitem_match = re.search(r"([가-하])\s*목", normalized)

    article_no = None
    if article_match:
        article_no = f"제{article_match.group(1)}조"
        if article_match.group(2):
            article_no += f"의{article_match.group(2)}"

    paragraph_no = f"제{paragraph_match.group(1)}항" if paragraph_match else None
    item_no = f"제{item_match.group(1)}호" if item_match else None
    subitem_no = f"{subitem_match.group(1)}목" if subitem_match else None
    return article_no, paragraph_no, item_no, subitem_no


def build_citations(
    parent_type: str,
    source_id: str,
    parent_id: str,
    texts: list[str],
) -> list[LegalCitation]:
    citations: list[LegalCitation] = []
    seen: set[str] = set()
    sequence_no = 1

    for text in texts:
        for citation_text in _extract_cites(text):
            key = citation_text
            if key in seen:
                continue
            seen.add(key)
            article_no, paragraph_no, item_no, subitem_no = _parse_citation_parts(citation_text)
            citations.append(
                LegalCitation(
                    citation_id=f"{parent_id}:cite:{sequence_no}",
                    source_id=source_id,
                    parent_type=parent_type,
                    parent_id=parent_id,
                    sequence_no=sequence_no,
                    citation_text=citation_text,
                    article_no=article_no,
                    paragraph_no=paragraph_no,
                    item_no=item_no,
                    subitem_no=subitem_no,
                )
            )
            sequence_no += 1

    return citations


def _first_cite(text: str) -> str | None:
    cites = _extract_cites(text)
    return cites[0] if cites else None


def _extract_effective_date(text: str) -> str | None:
    match = _EFFECTIVE_DATE_RE.search(text)
    if not match:
        return None
    value = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return value.isoformat()


def _extract_notice_no(text: str) -> str | None:
    match = _NOTICE_NO_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_limit(text: str) -> tuple[float | None, str | None]:
    normalized = _normalize_whitespace(text)
    if not any(keyword in normalized for keyword in _LIMIT_KEYWORDS):
        return None, None
    if not any(keyword in normalized for keyword in _TOTAL_KEYWORDS):
        return None, None
    for pattern, extractor in _PCT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            pct = extractor(match)
            if 0 < pct <= 1:
                return pct, normalized
    return None, None


def _source_type_from_content(name: str, text: str) -> str:
    normalized = _normalize_whitespace(text)
    if "항목별 사용 불가내역" in normalized:
        return "appendix_disallowed"
    if "질의회시집" in normalized or "해설집" in normalized or "해설" in name:
        return "commentary"
    return "law_notice"


def _clean_line(line: str) -> str:
    without_cite = _CITE_PATTERN.sub("", line).strip()
    without_cite = without_cite.replace("﻿", "")
    return without_cite


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in _NOISE_LINES:
        return True
    if any(stripped.startswith(prefix) for prefix in _NOISE_PREFIXES):
        return True
    return bool(_NOISE_RE.match(stripped))


def parse_source_documents(outputs_dir: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for final_md in sorted(outputs_dir.glob("*/final.md")):
        source_name = final_md.parent.name
        text = final_md.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        docs.append(
            SourceDocument(
                source_id=_slugify(source_name),
                source_name=source_name,
                source_type=_source_type_from_content(source_name, text),
                source_path=str(final_md.resolve()),
                title=title_match.group(1).strip() if title_match else source_name,
                effective_date=_extract_effective_date(text),
                notice_no=_extract_notice_no(text),
            )
        )
    return docs


def parse_legal_corpus(final_md: Path, source: SourceDocument) -> list[CorpusEntry]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    entries: list[CorpusEntry] = []
    heading_stack: dict[int, str] = {}
    current_context: str | None = None
    buffer_lines: list[str] = []
    buffer_cites: list[str] = []
    counter = 1

    def flush() -> None:
        nonlocal buffer_lines, buffer_cites, counter
        cleaned_lines = [_clean_line(line) for line in buffer_lines if not _is_noise_line(line)]
        cleaned_lines = [line for line in cleaned_lines if line]
        if not cleaned_lines:
            buffer_lines = []
            buffer_cites = []
            return

        body = "\n".join(cleaned_lines).strip()
        article_match = _ARTICLE_HEADER_RE.match(cleaned_lines[0].lstrip("- ").strip())
        article_no = f"제{article_match.group(1)}조" if article_match else None

        # 30자 미만 짧은 청크는 의미없는 노이즈로 제거 (실제 조문 제외)
        if len(body) < 30 and not article_no:
            buffer_lines = []
            buffer_cites = []
            return

        if source.source_type == "commentary":
            content_type = "commentary"
            # 해설서에서 【고시 제N조】 패턴으로 article_no 추출
            if not article_no:
                commentary_match = _COMMENTARY_ARTICLE_RE.search(body)
                if commentary_match:
                    article_no = commentary_match.group(1) or commentary_match.group(2)
                elif buffer_cites:
                    first_cite = buffer_cites[0]
                    cite_article = _ARTICLE_HEADER_RE.match(first_cite.strip())
                    if cite_article:
                        article_no = f"제{cite_article.group(1)}조"
        elif source.source_type == "appendix_disallowed":
            content_type = "appendix"
        elif article_no:
            content_type = "article"
        else:
            content_type = "section"

        title = None
        if article_match and article_match.group(2):
            title = article_match.group(2).strip()
        elif current_context:
            title = current_context.split(" > ")[-1]
        elif heading_stack:
            title = heading_stack[max(heading_stack)]

        entries.append(
            CorpusEntry(
                corpus_id=f"{source.source_id}:{counter:04d}",
                source_id=source.source_id,
                content_type=content_type,
                title=title,
                article_no=article_no,
                section_path=current_context or (" > ".join(heading_stack.values()) if heading_stack else None),
                body=body,
                cited_laws=buffer_cites[:],
                metadata={"source_type": source.source_type},
            )
        )
        counter += 1
        buffer_lines = []
        buffer_cites = []

    for line in lines:
        context_match = re.match(r"<!--\s*context:\s*(.+?)\s*-->", line)
        if context_match:
            current_context = context_match.group(1).strip()
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            for existing_level in list(heading_stack):
                if existing_level >= level:
                    del heading_stack[existing_level]
            heading_stack[level] = text
            continue

        if not line.strip():
            flush()
            continue

        if line.strip().startswith("|---") or line.strip().startswith("|---") or line.strip().startswith("|-"):
            continue

        buffer_lines.append(line)
        for cite in _extract_cites(line):
            if cite not in buffer_cites:
                buffer_cites.append(cite)

    flush()
    return entries


def parse_category_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    text = final_md.read_text(encoding="utf-8")
    start = text.find("제7조(사용기준)")
    end = text.find("② 제1항에도 불구하고", start)
    if start == -1:
        return []
    if end == -1:
        end = text.find("제8조(사용금액의 감액ㆍ반환 등)", start)
    if end == -1 or end <= start:
        return []

    block = text[start:end]
    block = re.sub(r"<!--\s*context:.*?-->\n?", "", block)
    block = re.sub(r"^법제처.*$", "", block, flags=re.MULTILINE)
    block = re.sub(r"^###\s+(\d+)\.\s+(.+)$", r"\1. \2", block, flags=re.MULTILINE)
    block = block.replace("비용 6. 근로자 건강장해예방비 등", "비용\n6. 근로자 건강장해예방비 등")

    rules: list[LegalRule] = []
    segment_pattern = re.compile(r"(?m)^(?:\[LEGAL_CITE:[^\]]+\]\s*)?([1-9])\.\s+")
    matches = list(segment_pattern.finditer(block))
    for idx, match in enumerate(matches):
        number = int(match.group(1))
        segment_start = match.start()
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        segment = block[segment_start:segment_end].strip()
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        if not lines:
            continue

        known_name = _CATEGORY_NAMES[number]
        legal_basis = _first_cite(segment)
        first_line_clean = _normalize_whitespace(_CITE_PATTERN.sub("", lines[0]))
        prefix = f"{number}. {known_name}"
        if first_line_clean.startswith(prefix):
            remainder = first_line_clean[len(prefix):].strip()
            rule_lines = ([remainder] if remainder else []) + [
                _normalize_whitespace(_CITE_PATTERN.sub("", line)) for line in lines[1:]
            ]
        else:
            first_line_without_number = re.sub(rf"^{number}\.\s*", "", first_line_clean)
            rule_lines = [first_line_without_number] + [
                _normalize_whitespace(_CITE_PATTERN.sub("", line)) for line in lines[1:]
            ]
        rule_lines = [line for line in rule_lines if line]
        rule_text = _clean_rule_text_for_storage(
            "\n".join(rule_lines).strip(),
            rule_type="category",
        )
        limit_pct, limit_rule_text = _extract_limit(rule_text)

        rules.append(
            LegalRule(
                rule_id=f"{source_id}:category:{number}",
                source_id=source_id,
                rule_type="category",
                category_code=_CATEGORY_CODES[number],
                category_number=number,
                category_name=known_name,
                allowed=True,
                keyword=known_name,
                item_pattern=None,
                legal_basis=legal_basis,
                limit_pct=limit_pct,
                rule_text=rule_text,
                metadata={"limit_rule_text": limit_rule_text or ""},
            )
        )
    return rules


def parse_disallowed_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    rules: list[LegalRule] = []
    for line in final_md.read_text(encoding="utf-8").splitlines():
        match = _APPENDIX_ROW_RE.match(line.strip())
        if not match:
            continue
        category_number = int(match.group(2))
        rules.append(
            LegalRule(
                rule_id=f"{source_id}:disallowed:{category_number}",
                source_id=source_id,
                rule_type="disallowed",
                category_code=_CATEGORY_CODES.get(category_number),
                category_number=category_number,
                category_name=_CATEGORY_NAMES.get(category_number, _normalize_whitespace(match.group(3))),
                allowed=False,
                keyword=None,
                item_pattern=None,
                legal_basis=_normalize_whitespace(match.group(1)),
                limit_pct=None,
                rule_text=_clean_rule_text_for_storage(
                    _normalize_whitespace(match.group(4)),
                    rule_type="disallowed",
                ),
                metadata={"raw_category_name": _normalize_whitespace(match.group(3))},
            )
        )

    if not any(rule.category_number == 7 for rule in rules):
        rules.append(
            LegalRule(
                rule_id=f"{source_id}:disallowed:7",
                source_id=source_id,
                rule_type="disallowed",
                category_code=_CATEGORY_CODES[7],
                category_number=7,
                category_name=_CATEGORY_NAMES[7],
                allowed=False,
                keyword=None,
                item_pattern=None,
                legal_basis=None,
                limit_pct=None,
                rule_text="-",
                metadata={"note": "원문 표에서 사용불가 내역이 '-'로 표기됨"},
            )
        )
    return rules


def parse_progress_appendix_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    progress_row: str | None = None
    usage_row: str | None = None
    appendix_seen = False

    for line in lines:
        stripped = line.strip()
        if "별표 3" in stripped:
            appendix_seen = True
            continue
        if not appendix_seen:
            continue
        if stripped.startswith("| 공정율") or stripped.startswith("| 공정률"):
            progress_row = stripped
            continue
        if progress_row and (stripped.startswith("| 사용기준") or stripped.startswith("| 사용 기준")):
            usage_row = stripped
            break

    if not progress_row or not usage_row:
        return []

    progress_cells = _split_markdown_row(progress_row)
    usage_cells = _split_markdown_row(usage_row)
    if len(progress_cells) <= 1 or len(usage_cells) <= 1:
        return []

    raw_table_text = "\n".join([progress_row, usage_row])
    rules: list[LegalRule] = []
    counter = 1

    for range_cell, usage_cell in zip(progress_cells[1:], usage_cells[1:]):
        range_info = _parse_progress_range_cell(range_cell)
        required_usage_rate = _parse_progress_usage_cell(usage_cell)
        if range_info is None or required_usage_rate is None:
            continue
        min_rate, max_rate = range_info
        if max_rate is None:
            range_text = f"공정률 {int(min_rate)}퍼센트 이상"
        else:
            range_text = f"공정률 {int(min_rate)}퍼센트 이상 {int(max_rate)}퍼센트 미만"
        rule_text = (
            f"{range_text} 구간에서는 산업안전보건관리비를 "
            f"{int(required_usage_rate * 100)}퍼센트 이상 사용하여야 한다."
        )
        rules.append(
            _make_rule(
                rule_id=f"{source_id}:progress:{counter}",
                source_id=source_id,
                rule_type="progress",
                category_number=None,
                allowed=True,
                legal_basis="별표 3",
                rule_text=rule_text,
                keyword="공정률 사용기준",
                item_pattern=range_text,
                metadata={
                    "source": "appendix_progress_table",
                    "appendix": "별표 3",
                    "min_progress_rate": min_rate,
                    "max_progress_rate": max_rate,
                    "required_usage_rate": required_usage_rate,
                    "raw_range_text": range_cell,
                    "raw_usage_text": usage_cell,
                    "raw_table_text": raw_table_text,
                },
            )
        )
        counter += 1

    return rules


def parse_appendix_1_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    in_appendix = False
    rules: list[LegalRule] = []
    construction_rows: list[tuple[str, str]] = []
    expected_names = ["건축공사", "토목공사", "중건설 공사", "특수건설 공사"]

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("【별표 1】"):
            in_appendix = True
            continue
        if in_appendix and stripped.startswith("【별표 1의2】"):
            break
        if not in_appendix or not stripped.startswith("|"):
            continue
        if "공사종류" in stripped or stripped.startswith("|----"):
            continue
        if any(token in stripped for token in ("건 | 축", "토 | 목", "중 | 건", "특 | 수")):
            construction_rows.append((stripped, ""))

    for idx, (row, _) in enumerate(construction_rows, start=1):
        values = _APPENDIX_1_VALUE_RE.findall(row)
        extracted: list[str] = []
        for pct, amount in values:
            if pct:
                extracted.append(f"{pct}%")
            elif amount:
                extracted.append(f"{amount}원")
        if len(extracted) < 5:
            continue
        name = expected_names[idx - 1] if idx - 1 < len(expected_names) else f"공사종류 {idx}"
        under_5, between_5_50, base_amount, over_50, manager_rate = extracted[:5]
        rule_text = (
            f"{name}의 산업안전보건관리비 계상기준은 대상액 5억 원 미만 {under_5}, "
            f"5억 원 이상 50억 원 미만 적용비율 {between_5_50}와 기초액 {base_amount}, "
            f"50억 원 이상 {over_50}, 보건관리자 선임 대상 공사 {manager_rate}이다."
        )
        rules.append(
            _make_rule(
                rule_id=f"{source_id}:appendix1:{idx}",
                source_id=source_id,
                rule_type="qa_limit",
                category_number=None,
                allowed=True,
                legal_basis="별표 1",
                rule_text=rule_text,
                keyword=name,
                item_pattern=name,
                metadata={
                    "source": "appendix_1_table",
                    "appendix": "별표 1",
                    "construction_type": name,
                    "under_5b_rate_pct": under_5,
                    "between_5b_50b_rate_pct": between_5_50,
                    "between_5b_50b_base_amount": base_amount,
                    "over_50b_rate_pct": over_50,
                    "manager_target_rate_pct": manager_rate,
                    "raw_row_text": row,
                },
            )
        )
    return rules


def parse_appendix_1_2_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    in_appendix = False
    items: list[str] = []
    current: str | None = None

    for line in lines:
        stripped = _normalize_whitespace(line)
        if stripped.startswith("【별표 1의2】"):
            in_appendix = True
            continue
        if in_appendix and stripped.startswith("【별표 1의3】"):
            break
        if not in_appendix or not stripped:
            continue
        match = _APPENDIX_LIST_ITEM_RE.match(stripped)
        if match:
            content = _normalize_whitespace(match.group(2))
            if "파쇄에 한정한다" in content and current:
                current = f"{current} {content}"
                continue
            if current:
                items.append(current)
            current = content
            continue
        if current:
            current = f"{current} {stripped}"
    if current:
        items.append(current)

    rules: list[LegalRule] = []
    for idx, item in enumerate(items, start=1):
        rules.append(
            _make_rule(
                rule_id=f"{source_id}:appendix1-2:{idx}",
                source_id=source_id,
                rule_type="allowed",
                category_number=1,
                allowed=True,
                legal_basis="별표 1의2",
                rule_text=f"관리감독자 안전보건업무 수행 시 수당지급 작업에 해당한다: {item}",
                keyword="관리감독자 수당지급 작업",
                item_pattern=item,
                metadata={
                    "source": "appendix_1_2_list",
                    "appendix": "별표 1의2",
                    "task_index": idx,
                },
            )
        )
    return rules


def parse_appendix_1_3_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    in_appendix = False
    current_number: str | None = None
    current_text: str | None = None
    current_formula: str | None = None
    collected: list[tuple[str, str, str]] = []

    for line in lines:
        stripped = _normalize_whitespace(line)
        if stripped.startswith("【별표 1의3】"):
            in_appendix = True
            continue
        if in_appendix and "【별표 2】" in stripped:
            break
        if not in_appendix or not stripped:
            continue
        match = _APPENDIX_LIST_ITEM_RE.match(stripped)
        if match:
            if current_number and current_text and current_formula:
                collected.append((current_number, current_text, current_formula))
            current_number = match.group(1)
            current_text = _normalize_whitespace(match.group(2))
            current_formula = None
            continue
        if stripped.startswith("-") and current_number:
            current_formula = _normalize_whitespace(stripped.lstrip("- "))
            continue
        if current_text and current_number and current_formula is None:
            current_text = f"{current_text} {stripped}"
        elif current_formula:
            current_formula = f"{current_formula} {stripped}"
    if current_number and current_text and current_formula:
        collected.append((current_number, current_text, current_formula))

    formula_types = {
        "1": "adjusted_amount_formula",
        "2": "delta_formula",
        "3": "change_ratio_formula",
    }
    rules: list[LegalRule] = []
    for number, text, formula in collected:
        rules.append(
            _make_rule(
                rule_id=f"{source_id}:appendix1-3:{number}",
                source_id=source_id,
                rule_type="qa_limit",
                category_number=None,
                allowed=True,
                legal_basis="별표 1의3",
                rule_text=f"{text} {formula}",
                keyword="설계변경 조정계상",
                item_pattern=text,
                metadata={
                    "source": "appendix_1_3_formula",
                    "appendix": "별표 1의3",
                    "formula_index": int(number),
                    "formula_type": formula_types.get(number, "formula"),
                    "formula_text": formula,
                },
            )
        )
    return rules


def _make_rule(
    *,
    rule_id: str,
    source_id: str,
    rule_type: str,
    category_number: int | None,
    allowed: bool | None,
    legal_basis: str | None,
    rule_text: str,
    keyword: str | None = None,
    item_pattern: str | None = None,
    limit_pct: float | None = None,
    metadata: dict | None = None,
) -> LegalRule:
    category_code = _CATEGORY_CODES.get(category_number) if category_number else None
    category_name = _CATEGORY_NAMES.get(category_number) if category_number else None
    cleaned_rule_text = _clean_rule_text_for_storage(rule_text, rule_type=rule_type)
    return LegalRule(
        rule_id=rule_id,
        source_id=source_id,
        rule_type=rule_type,
        category_code=category_code,
        category_number=category_number,
        category_name=category_name,
        allowed=allowed,
        keyword=keyword,
        item_pattern=item_pattern,
        legal_basis=legal_basis,
        limit_pct=limit_pct,
        rule_text=cleaned_rule_text or rule_text,
        metadata=metadata or {},
    )


def parse_law_detail_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    raw_lines = final_md.read_text(encoding="utf-8").splitlines()

    # Pre-process: "- 바. ... 비용 6. 근로자 건강장해예방비 등" → split into two lines.
    # The OCR merged the CAT_05 last item with the CAT_06 section header.
    lines: list[str] = []
    for raw in raw_lines:
        if "비용 6. 근로자 건강장해예방비 등" in raw:
            idx = raw.find("6. 근로자 건강장해예방비 등")
            lines.append(raw[:idx].rstrip())
            lines.append("6. 근로자 건강장해예방비 등")
        else:
            lines.append(raw)

    rules: list[LegalRule] = []
    in_article7 = False
    in_exclusion = False
    current_category: int | None = None
    counter = 1

    # Buffer for merging OCR-split continuation lines into a single rule.
    buf_text: str | None = None
    buf_basis: str | None = None
    buf_cat: int | None = None

    def _flush() -> None:
        nonlocal buf_text, buf_basis, buf_cat, counter
        if buf_text is None or buf_cat is None:
            return
        cleaned_buf_text = _clean_rule_text_for_storage(buf_text, rule_type="allowed")
        limit_pct, limit_rule_text = _extract_limit(cleaned_buf_text)
        base_id = f"{source_id}:law-detail:{counter}"
        rules.append(_make_rule(
            rule_id=base_id, source_id=source_id, rule_type="allowed",
            category_number=buf_cat, allowed=True, legal_basis=buf_basis,
            rule_text=cleaned_buf_text, keyword=_CATEGORY_NAMES[buf_cat],
            item_pattern=cleaned_buf_text, metadata={"source": "article7_detail"},
        ))
        counter += 1
        if limit_pct is not None:
            rules.append(_make_rule(
                rule_id=f"{base_id}:limit", source_id=source_id, rule_type="limit",
                category_number=buf_cat, allowed=True, legal_basis=buf_basis,
                rule_text=limit_rule_text or cleaned_buf_text, keyword=_CATEGORY_NAMES[buf_cat],
                item_pattern=cleaned_buf_text, limit_pct=limit_pct,
                metadata={"source": "article7_limit"},
            ))
            counter += 1
        buf_text = buf_basis = buf_cat = None

    for raw_line in lines:
        line = raw_line.strip()
        if "제7조(사용기준)" in line:
            in_article7 = True
            continue
        if not in_article7:
            continue
        if "제8조(사용금액의 감액ㆍ반환 등)" in line:
            break
        if not line or line.startswith("법제처"):
            continue

        heading_match = re.match(r"^###\s+(\d+)\.\s+(.+)$", line)
        if heading_match:
            _flush()
            current_category = int(heading_match.group(1))
            in_exclusion = False
            continue

        if "② 제1항에도 불구하고" in line:
            _flush()
            in_exclusion = True
            current_category = None
            continue

        cleaned = _normalize_whitespace(_CITE_PATTERN.sub("", line))
        legal_basis = _first_cite(line)
        if not cleaned:
            continue

        # Detect inline category headers for 6-9 (no ### heading in markdown).
        # Category 6 header was split from the prev line in pre-processing.
        # Categories 7-9 carry their full rule content on the numbered line.
        inline_cat = re.match(r"^([6-9])\.", cleaned)
        if inline_cat:
            new_cat = int(inline_cat.group(1))
            _flush()
            current_category = new_cat
            in_exclusion = False
            if new_cat == 6:
                continue  # pure section header; actual items follow as bullets

        if in_exclusion:
            if re.match(r"^[1-4]\.", cleaned):
                rules.append(_make_rule(
                    rule_id=f"{source_id}:law-exclusion:{counter}",
                    source_id=source_id, rule_type="disallowed",
                    category_number=None, allowed=False,
                    legal_basis=legal_basis,
                    rule_text=_clean_rule_text_for_storage(cleaned, rule_type="disallowed"),
                    item_pattern=_clean_rule_text_for_storage(cleaned, rule_type="disallowed"),
                    metadata={"source": "article7_exclusion"},
                ))
                counter += 1
            continue

        if current_category is None:
            continue

        # A line starts a new rule item if it begins with 가./나./.../바. (with or
        # without a leading "- ") or is a numbered category line (7/8/9).
        content = cleaned[2:].strip() if cleaned.startswith("- ") else cleaned
        is_new_item = (
            bool(re.match(r"^[가나다라마바]\.\s", content))
            or cleaned.startswith(("가.", "나.", "다.", "라.", "마.", "바."))
            or bool(inline_cat)
        )
        # A continuation line starts with "- " but is NOT a new sub-item — it is
        # the second half of an OCR-split sentence from the previous item.
        is_continuation = (
            buf_text is not None
            and buf_cat == current_category
            and cleaned.startswith("- ")
            and not is_new_item
        )

        if is_continuation:
            buf_text = buf_text + " " + content
        elif is_new_item or cleaned.startswith("-"):
            _flush()
            buf_text = cleaned
            buf_basis = legal_basis
            buf_cat = current_category

    _flush()
    return rules


def _infer_allowed_from_answer(text: str) -> tuple[bool | None, str]:
    normalized = _normalize_whitespace(text)
    has_allow = any(token in normalized for token in ["사용이 가능", "사용 가능", "가능함", "가능할 것"])
    has_disallow = any(token in normalized for token in ["사용이 불가", "사용 불가", "불가함", "불가할", "제외"])
    if has_allow and not has_disallow:
        return True, "allow_only"
    if has_disallow and not has_allow:
        return False, "disallow_only"
    if has_allow and has_disallow:
        return True, "mixed_with_exception"
    return None, "undetermined"


def _is_rule_like_text(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if len(normalized) < 10:
        return False
    if normalized.startswith("|"):
        return False
    if any(token in normalized for token in _REGULATORY_TOKENS):
        return True
    return bool(_extract_cites(normalized))


def _infer_rule_type(text: str) -> tuple[str, bool | None, float | None]:
    limit_pct, _ = _extract_limit(text)
    if limit_pct is not None:
        return "rule_like_limit", True, limit_pct

    normalized = _normalize_whitespace(text)
    has_allow = any(token in normalized for token in ["사용이 가능", "사용 가능", "가능함", "할 수 있다"])
    has_disallow = any(token in normalized for token in ["사용이 불가", "사용 불가", "불가함", "할 수 없다", "초과할 수 없다"])

    if has_allow and not has_disallow:
        return "rule_like_allowed", True, None
    if has_disallow and not has_allow:
        return "rule_like_disallowed", False, None
    return "rule_like", None, None


def parse_rule_like_corpus_rules(corpus_entries: list[CorpusEntry]) -> list[LegalRule]:
    rules: list[LegalRule] = []
    counter = 1

    for entry in corpus_entries:
        if not _is_rule_like_text(entry.body):
            continue

        rule_type, allowed, limit_pct = _infer_rule_type(entry.body)
        cleaned_entry_body = _clean_rule_text_for_storage(entry.body, rule_type=rule_type)
        if _is_question_like_rule_text(cleaned_entry_body):
            continue
        category_number = None
        for number, name in _CATEGORY_NAMES.items():
            if entry.section_path and name in entry.section_path:
                category_number = number
                break
            if entry.title and name in entry.title:
                category_number = number
                break

        rules.append(
            _make_rule(
                rule_id=f"{entry.source_id}:corpus-rule:{counter}",
                source_id=entry.source_id,
                rule_type=rule_type,
                category_number=category_number,
                allowed=allowed,
                legal_basis=entry.cited_laws[0] if entry.cited_laws else None,
                rule_text=cleaned_entry_body,
                keyword=entry.title,
                item_pattern=cleaned_entry_body[:200],
                limit_pct=limit_pct,
                metadata={
                    "source": "corpus_rule_like",
                    "corpus_id": entry.corpus_id,
                    "content_type": entry.content_type,
                    "section_path": entry.section_path or "",
                },
            )
        )
        counter += 1

    return rules


def build_verification_report(documents: list[SourceDocument], corpus_entries: list[CorpusEntry]) -> dict:
    report: dict[str, dict] = {}
    for source in documents:
        source_text = Path(source.source_path).read_text(encoding="utf-8")
        source_lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip() and not line.strip().startswith("<!--")
        ]
        entry_bodies = [
            entry.body for entry in corpus_entries if entry.source_id == source.source_id
        ]
        covered_entries = len(entry_bodies)
        cited_count = sum(len(entry.cited_laws) for entry in corpus_entries if entry.source_id == source.source_id)
        report[source.source_id] = {
            "source_name": source.source_name,
            "source_type": source.source_type,
            "nonempty_source_lines": len(source_lines),
            "corpus_entries": covered_entries,
            "cited_laws_found": cited_count,
            "has_corpus_entries": covered_entries > 0,
            "coverage_note": "All markdown files were parsed into legal_corpus entries; inspect corpus_entries vs source lines for granularity.",
        }
    return report


def parse_rule_profiles(rule_config_path: Path) -> list[LegalRuleProfile]:
    if not rule_config_path.exists():
        return []

    config = json.loads(rule_config_path.read_text(encoding="utf-8"))
    profiles: list[LegalRuleProfile] = []

    for synonym_key, values in config.get("validator_synonyms", {}).items():
        profiles.append(
            LegalRuleProfile(
                profile_id=f"validator_synonym:{_slugify(synonym_key)}",
                profile_scope="validator_synonym",
                category_code=None,
                profile_key=synonym_key,
                values_json=values,
                metadata={"source": str(rule_config_path.resolve())},
            )
        )

    for category_code, profile in config.get("validator_profiles", {}).items():
        for profile_key, values in profile.items():
            profiles.append(
                LegalRuleProfile(
                    profile_id=f"validator_profile:{category_code}:{profile_key}",
                    profile_scope="validator_profile",
                    category_code=category_code,
                    profile_key=profile_key,
                    values_json=values,
                    metadata={"source": str(rule_config_path.resolve())},
                )
            )

    for category_code, profile in config.get("classifier_profiles", {}).items():
        for profile_key, values in profile.items():
            profiles.append(
                LegalRuleProfile(
                    profile_id=f"classifier_profile:{category_code}:{profile_key}",
                    profile_scope="classifier_profile",
                    category_code=category_code,
                    profile_key=profile_key,
                    values_json=values,
                    metadata={"source": str(rule_config_path.resolve())},
                )
            )

    for item_key, policy in config.get("generic_item_policies", {}).items():
        conditional_categories = policy.get("conditional_categories") or []
        category_code = conditional_categories[0] if len(conditional_categories) == 1 else None
        profiles.append(
            LegalRuleProfile(
                profile_id=f"generic_item_policy:{_slugify(item_key)}",
                profile_scope="generic_item_policy",
                category_code=category_code,
                profile_key=item_key,
                values_json=policy,
                metadata={"source": str(rule_config_path.resolve())},
            )
        )

    return profiles


def _infer_category_from_text(*, title: str | None = None, section_path: str | None = None, body: str | None = None) -> tuple[str | None, str | None]:
    for number, name in _CATEGORY_NAMES.items():
        if title and name in title:
            return _CATEGORY_CODES[number], name
        if section_path and name in section_path:
            return _CATEGORY_CODES[number], name
        if body and name in body:
            return _CATEGORY_CODES[number], name
    return None, None


def build_master_rows(
    *,
    documents: list[SourceDocument],
    corpus_entries: list[CorpusEntry],
    rule_entries: list[LegalRule],
    rule_profiles: list[LegalRuleProfile],
) -> list[dict]:
    source_map = {doc.source_id: doc for doc in documents}
    master_rows: list[dict] = []

    for entry in corpus_entries:
        source = source_map[entry.source_id]
        category_code, category_name = _infer_category_from_text(
            title=entry.title,
            section_path=entry.section_path,
            body=entry.body,
        )
        master_rows.append(
            {
                "master_id": f"master:corpus:{entry.corpus_id}",
                "source_id": entry.source_id,
                "source_name": source.source_name,
                "source_type": source.source_type,
                "record_type": "corpus",
                "content_type": entry.content_type,
                "rule_type": None,
                "profile_scope": None,
                "category_code": category_code,
                "category_name": category_name,
                "article_no": entry.article_no,
                "title": entry.title,
                "section_path": entry.section_path,
                "legal_basis": entry.cited_laws[0] if entry.cited_laws else None,
                "item_key": None,
                "item_pattern": None,
                "allowed": None,
                "limit_pct": None,
                "body": entry.body,
                "cited_laws": entry.cited_laws,
                "keywords": [value for value in [entry.title, entry.article_no] if value],
                "metadata": {"source_record": "legal_corpus", **entry.metadata},
            }
        )

    for rule in rule_entries:
        source = source_map[rule.source_id]
        master_rows.append(
            {
                "master_id": f"master:rule:{rule.rule_id}",
                "source_id": rule.source_id,
                "source_name": source.source_name,
                "source_type": source.source_type,
                "record_type": "rule",
                "content_type": None,
                "rule_type": rule.rule_type,
                "profile_scope": None,
                "category_code": rule.category_code,
                "category_name": rule.category_name,
                "article_no": None,
                "title": rule.category_name or rule.keyword,
                "section_path": None,
                "legal_basis": rule.legal_basis,
                "item_key": rule.keyword,
                "item_pattern": rule.item_pattern,
                "allowed": rule.allowed,
                "limit_pct": rule.limit_pct,
                "body": rule.rule_text,
                "cited_laws": [rule.legal_basis] if rule.legal_basis else [],
                "keywords": [value for value in [rule.keyword, rule.item_pattern] if value],
                "metadata": {"source_record": "legal_rules", **rule.metadata},
            }
        )

    for profile in rule_profiles:
        master_rows.append(
            {
                "master_id": f"master:profile:{profile.profile_id}",
                "source_id": None,
                "source_name": "scripts/seed_legal_rule_profiles.json",
                "source_type": "rule_config",
                "record_type": "profile",
                "content_type": None,
                "rule_type": None,
                "profile_scope": profile.profile_scope,
                "category_code": profile.category_code,
                "category_name": None,
                "article_no": None,
                "title": profile.profile_key,
                "section_path": None,
                "legal_basis": None,
                "item_key": profile.profile_key,
                "item_pattern": None,
                "allowed": None,
                "limit_pct": None,
                "body": json.dumps(profile.values_json, ensure_ascii=False),
                "cited_laws": [],
                "keywords": [profile.profile_key],
                "metadata": {"source_record": "legal_rule_profiles", **profile.metadata},
            }
        )

    return master_rows


def parse_commentary_qa_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    lines = final_md.read_text(encoding="utf-8").splitlines()
    rules: list[LegalRule] = []
    current_category: int | None = None
    current_question: str | None = None
    current_answer_lines: list[str] = []
    current_context: str | None = None
    counter = 1

    def flush() -> None:
        nonlocal current_question, current_answer_lines, counter
        if not current_question or not current_answer_lines:
            current_question = None
            current_answer_lines = []
            return
        answer_text = "\n".join(
            _normalize_whitespace(_CITE_PATTERN.sub("", line))
            for line in current_answer_lines
            if _normalize_whitespace(_CITE_PATTERN.sub("", line))
        ).strip()
        if not answer_text:
            current_question = None
            current_answer_lines = []
            return

        cleaned_answer_text = _clean_rule_text_for_storage(answer_text, rule_type="qa")
        if _is_question_like_rule_text(cleaned_answer_text):
            current_question = None
            current_answer_lines = []
            return

        allowed, mode = _infer_allowed_from_answer(cleaned_answer_text)
        limit_pct, limit_rule_text = _extract_limit(cleaned_answer_text)
        cleaned_limit_rule_text = _clean_rule_text_for_storage(limit_rule_text or "", rule_type="qa") if limit_rule_text else ""
        legal_basis = None
        for line in current_answer_lines:
            legal_basis = _first_cite(line)
            if legal_basis:
                break

        rule_type = "qa"
        if limit_pct is not None:
            rule_type = "qa_limit"
        elif allowed is False:
            rule_type = "qa_disallowed"
        elif allowed is True:
            rule_type = "qa_allowed"

        rules.append(
            _make_rule(
                rule_id=f"{source_id}:qa:{counter}",
                source_id=source_id,
                rule_type=rule_type,
                category_number=current_category,
                allowed=allowed,
                legal_basis=legal_basis,
                rule_text=cleaned_answer_text,
                keyword=current_question,
                item_pattern=current_question,
                limit_pct=limit_pct,
                metadata={
                    "source": "commentary_qa",
                    "question": current_question,
                    "context": current_context or "",
                    "inference_mode": mode,
                    "limit_rule_text": cleaned_limit_rule_text,
                },
            )
        )
        counter += 1
        current_question = None
        current_answer_lines = []

    for raw_line in lines:
        context_match = re.match(r"<!--\s*context:\s*(.+?)\s*-->", raw_line)
        if context_match:
            current_context = context_match.group(1).strip()
            continue

        heading_match = re.match(r"^###\s+(\d+)\.\s+(.+)$", raw_line.strip())
        if heading_match:
            flush()
            current_category = int(heading_match.group(1))
            continue

        question_match = _QUESTION_PATTERN.match(raw_line.strip())
        if question_match and "사용 가능한지" in question_match.group(2):
            flush()
            current_question = _normalize_whitespace(question_match.group(2))
            current_answer_lines = []
            continue

        if current_question:
            if raw_line.strip().startswith("(건설산재예방정책과") or raw_line.strip().startswith("(2024년") or raw_line.strip().startswith("(2025년"):
                flush()
                continue
            current_answer_lines.append(raw_line)

    flush()
    return rules


def build_payload(outputs_dir: Path, rule_config_path: Path = Path("scripts/seed_legal_rule_profiles.json")) -> dict:
    documents = parse_source_documents(outputs_dir)
    corpus: list[dict] = []
    corpus_entries: list[CorpusEntry] = []
    for source in documents:
        entries = parse_legal_corpus(Path(source.source_path), source)
        corpus_entries.extend(entries)
        corpus.extend(asdict(entry) for entry in entries)

    law_doc = next(doc for doc in documents if doc.source_type == "law_notice")
    appendix_doc = next(doc for doc in documents if doc.source_type == "appendix_disallowed")
    commentary_doc = next(doc for doc in documents if doc.source_type == "commentary")
    progress_docs = [
        doc for doc in documents
        if "건설업 산업안전 보건관리비 해설 및 질의회시집" in doc.source_name
        or "별표 3" in Path(doc.source_path).read_text(encoding="utf-8")
    ]
    progress_rules: list[LegalRule] = []
    for progress_doc in progress_docs:
        progress_rules = parse_progress_appendix_rules(Path(progress_doc.source_path), progress_doc.source_id)
        if progress_rules:
            break
    appendix_1_rules = parse_appendix_1_rules(Path(commentary_doc.source_path), commentary_doc.source_id)
    appendix_1_2_rules = parse_appendix_1_2_rules(Path(commentary_doc.source_path), commentary_doc.source_id)
    appendix_1_3_rules = parse_appendix_1_3_rules(Path(commentary_doc.source_path), commentary_doc.source_id)

    rule_entries = [
        *parse_category_rules(Path(law_doc.source_path), law_doc.source_id),
        *parse_law_detail_rules(Path(law_doc.source_path), law_doc.source_id),
        *parse_disallowed_rules(Path(appendix_doc.source_path), appendix_doc.source_id),
        *appendix_1_rules,
        *appendix_1_2_rules,
        *appendix_1_3_rules,
        *progress_rules,
        *parse_commentary_qa_rules(Path(commentary_doc.source_path), commentary_doc.source_id),
        *parse_rule_like_corpus_rules(corpus_entries),
    ]
    rules = [asdict(rule) for rule in rule_entries]

    citations: list[dict] = []
    for entry in corpus_entries:
        citations.extend(
            asdict(citation)
            for citation in build_citations(
                parent_type="corpus",
                source_id=entry.source_id,
                parent_id=entry.corpus_id,
                texts=[entry.body],
            )
        )

    for rule in rule_entries:
        rule_texts = [rule.rule_text]
        if rule.legal_basis:
            rule_texts.insert(0, f"[LEGAL_CITE: {rule.legal_basis}]")
        citations.extend(
            asdict(citation)
            for citation in build_citations(
                parent_type="rule",
                source_id=rule.source_id,
                parent_id=rule.rule_id,
                texts=rule_texts,
            )
        )

    verification = build_verification_report(documents, corpus_entries)
    parsed_rule_profiles = parse_rule_profiles(rule_config_path)
    rule_profiles = [asdict(profile) for profile in parsed_rule_profiles]
    master_rows = build_master_rows(
        documents=documents,
        corpus_entries=corpus_entries,
        rule_entries=rule_entries,
        rule_profiles=parsed_rule_profiles,
    )

    return {
        "documents": [asdict(doc) for doc in documents],
        "corpus": corpus,
        "rules": rules,
        "citations": citations,
        "rule_profiles": rule_profiles,
        "master": master_rows,
        "verification": verification,
    }


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False).replace("'", "''")
        return f"'{text}'::jsonb"
    if isinstance(value, list):
        escaped = ", ".join(_sql_literal(item) for item in value)
        suffix = "::text[]" if not value else ""
        return f"ARRAY[{escaped}]{suffix}"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def payload_to_sql(payload: dict) -> str:
    statements = ["BEGIN;", "SET LOCAL search_path TO legal_rag, public;"]
    statements.append(
        "TRUNCATE legal_rule_master, legal_rule_profiles, legal_citations, legal_rules, legal_corpus, legal_source_documents RESTART IDENTITY CASCADE;"
    )

    for row in payload["documents"]:
        v2_st = _V2_SOURCE_TYPE.get(row["source_type"], row["source_type"])
        statements.append(
            "INSERT INTO legal_source_documents "
            "(source_id, source_name, source_type, source_path, title, effective_date, notice_no) VALUES "
            f"({_sql_literal(row['source_id'])}, {_sql_literal(row['source_name'])}, {_sql_literal(v2_st)}, "
            f"{_sql_literal(row['source_path'])}, {_sql_literal(row['title'])}, {_sql_literal(row['effective_date'])}, {_sql_literal(row['notice_no'])});"
        )

    for row in payload["corpus"]:
        original_ct = row["content_type"]
        v2_ct = _V2_CONTENT_TYPE.get(original_ct, original_ct)
        metadata = {**row["metadata"], **({"original_content_type": original_ct} if v2_ct != original_ct else {})}
        statements.append(
            "INSERT INTO legal_corpus "
            "(corpus_id, source_id, content_type, title, article_no, section_path, body, cited_laws, metadata) VALUES "
            f"({_sql_literal(row['corpus_id'])}, {_sql_literal(row['source_id'])}, {_sql_literal(v2_ct)}, "
            f"{_sql_literal(row['title'])}, {_sql_literal(row['article_no'])}, {_sql_literal(row['section_path'])}, "
            f"{_sql_literal(row['body'])}, {_sql_literal(row['cited_laws'])}, {_sql_literal(metadata)});"
        )

    for row in payload["rules"]:
        original_rt = row["rule_type"]
        v2_rt = _V2_RULE_TYPE.get(original_rt, original_rt)
        metadata = {**row["metadata"], **({"original_rule_type": original_rt} if v2_rt != original_rt else {})}
        statements.append(
            "INSERT INTO legal_rules "
            "(rule_id, source_id, rule_type, category_code, category_number, category_name, allowed, keyword, item_pattern, legal_basis, limit_pct, rule_text, metadata) VALUES "
            f"({_sql_literal(row['rule_id'])}, {_sql_literal(row['source_id'])}, {_sql_literal(v2_rt)}, "
            f"{_sql_literal(row['category_code'])}, {_sql_literal(row['category_number'])}, {_sql_literal(row['category_name'])}, "
            f"{_sql_literal(row['allowed'])}, {_sql_literal(row['keyword'])}, {_sql_literal(row['item_pattern'])}, "
            f"{_sql_literal(row['legal_basis'])}, {_sql_literal(row['limit_pct'])}, {_sql_literal(row['rule_text'])}, {_sql_literal(metadata)});"
        )

    for row in payload["citations"]:
        statements.append(
            "INSERT INTO legal_citations "
            "(citation_id, source_id, parent_type, parent_id, sequence_no, citation_text, article_no, paragraph_no, item_no, subitem_no) VALUES "
            f"({_sql_literal(row['citation_id'])}, {_sql_literal(row['source_id'])}, {_sql_literal(row['parent_type'])}, "
            f"{_sql_literal(row['parent_id'])}, {_sql_literal(row['sequence_no'])}, {_sql_literal(row['citation_text'])}, "
            f"{_sql_literal(row['article_no'])}, {_sql_literal(row['paragraph_no'])}, {_sql_literal(row['item_no'])}, {_sql_literal(row['subitem_no'])});"
        )

    for row in payload.get("rule_profiles", []):
        original_scope = row["profile_scope"]
        v2_scope = _V2_PROFILE_SCOPE.get(original_scope, original_scope)
        metadata = {**row["metadata"], **({"original_scope": original_scope} if v2_scope != original_scope else {})}
        vj = row["values_json"]
        vj_sql = _sql_literal(vj) if isinstance(vj, dict) else "'" + json.dumps(vj, ensure_ascii=False).replace("'", "''") + "'::jsonb"
        statements.append(
            "INSERT INTO legal_rule_profiles "
            "(profile_id, profile_scope, category_code, profile_key, values_json, metadata) VALUES "
            f"({_sql_literal(row['profile_id'])}, {_sql_literal(v2_scope)}, {_sql_literal(row['category_code'])}, "
            f"{_sql_literal(row['profile_key'])}, {vj_sql}, {_sql_literal(metadata)});"
        )

    for row in payload.get("master", []):
        original_ct = row.get("content_type")
        v2_ct = _V2_CONTENT_TYPE.get(original_ct, original_ct) if original_ct else None
        original_rt = row.get("rule_type")
        v2_rt = _V2_RULE_TYPE.get(original_rt, original_rt) if original_rt else None
        original_ps = row.get("profile_scope")
        v2_ps = _V2_PROFILE_SCOPE.get(original_ps, original_ps) if original_ps else None
        original_st = row.get("source_type")
        v2_st = _V2_SOURCE_TYPE.get(original_st, original_st) if original_st else None
        metadata = dict(row["metadata"])
        if v2_ct != original_ct and original_ct:
            metadata["original_content_type"] = original_ct
        if v2_rt != original_rt and original_rt:
            metadata["original_rule_type"] = original_rt
        if v2_ps != original_ps and original_ps:
            metadata["original_scope"] = original_ps
        statements.append(
            "INSERT INTO legal_rule_master "
            "(master_id, source_id, source_name, source_type, record_type, content_type, rule_type, profile_scope, "
            "category_code, category_name, article_no, title, section_path, legal_basis, item_key, item_pattern, "
            "allowed, limit_pct, body, cited_laws, keywords, metadata) VALUES "
            f"({_sql_literal(row['master_id'])}, {_sql_literal(row['source_id'])}, {_sql_literal(row['source_name'])}, "
            f"{_sql_literal(v2_st)}, {_sql_literal(row['record_type'])}, {_sql_literal(v2_ct)}, "
            f"{_sql_literal(v2_rt)}, {_sql_literal(v2_ps)}, {_sql_literal(row['category_code'])}, "
            f"{_sql_literal(row['category_name'])}, {_sql_literal(row['article_no'])}, {_sql_literal(row['title'])}, "
            f"{_sql_literal(row['section_path'])}, {_sql_literal(row['legal_basis'])}, {_sql_literal(row['item_key'])}, "
            f"{_sql_literal(row['item_pattern'])}, {_sql_literal(row['allowed'])}, {_sql_literal(row['limit_pct'])}, "
            f"{_sql_literal(row['body'])}, {_sql_literal(row['cited_laws'])}, {_sql_literal(row['keywords'])}, {_sql_literal(metadata)});"
        )

    statements.append("COMMIT;")
    return "\n".join(statements) + "\n"


def _run_psql(sql_path: Path, database_url: str | None, psql_bin: str) -> None:
    cmd = [psql_bin]
    if database_url:
        cmd.append(database_url)
    cmd.extend(["-v", "ON_ERROR_STOP=1", "-f", str(sql_path)])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract legal corpus and decision rules, then load them into PostgreSQL."
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--rule-config-path", default="scripts/seed_legal_rule_profiles.json")
    parser.add_argument("--json-out", default="artifacts/legal_rules_payload.json")
    parser.add_argument("--sql-out", default="artifacts/legal_rules_seed.sql")
    parser.add_argument("--apply", action="store_true", help="Seed SQL을 legal_rag 스키마에 적재한다 (스키마는 Flyway V2가 생성).")
    parser.add_argument("--database-url", default=None, help="PostgreSQL connection string for psql.")
    parser.add_argument("--psql-bin", default="psql")
    parser.add_argument("--cleanup", action="store_true", help="DB 적재 성공 후 중간 파일(json/sql) 삭제.")
    args = parser.parse_args()

    payload = build_payload(Path(args.outputs_dir), Path(args.rule_config_path))

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sql_out = Path(args.sql_out)
    sql_out.parent.mkdir(parents=True, exist_ok=True)
    sql_out.write_text(payload_to_sql(payload), encoding="utf-8")

    if args.apply:
        _run_psql(sql_out, args.database_url, args.psql_bin)
        if args.cleanup:
            json_out.unlink(missing_ok=True)
            sql_out.unlink(missing_ok=True)
            print("cleanup: 중간 파일 삭제 완료")

    print(f"documents={len(payload['documents'])}")
    print(f"corpus={len(payload['corpus'])}")
    print(f"rules={len(payload['rules'])}")
    print(f"citations={len(payload['citations'])}")
    print(f"rule_profiles={len(payload.get('rule_profiles', []))}")
    print(f"master={len(payload.get('master', []))}")
    print(f"verified_sources={len(payload['verification'])}")


if __name__ == "__main__":
    main()
