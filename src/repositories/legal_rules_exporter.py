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
# TOC(목차)형 행: 점선으로 이어진 목차 패턴 (예: "| 01 해설집···· · · · ·")
_TOC_LINE_RE = re.compile(
    r"[·\.·]{5,}"                            # 연속 점 5개 이상 (·, ., ·)
    r"|\|\s*\d{1,3}\s*\|?\s*$"                    # | 숫자 | 로 끝나는 페이지 참조
)
# OCR 띄어짐 보정 대상: 한글 한 글자 + 공백 + 한글 한 글자 패턴이 3회 이상
_OCR_SPLIT_RE = re.compile(r"([가-힣])\s([가-힣])\s([가-힣])")
# 영문 단독 쓰레기 행 (의미 없는 짧은 영문)
_ENGLISH_JUNK_RE = re.compile(r"^[A-Za-z\s\.\,\-]{1,40}$")
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
# --------------------------------------------------------------------------
# rule_type 정규화 방침
#   DB 저장 타입: allowed | disallowed | limit | category | progress | qa
#   의미 구분은 metadata.source_kind 로 보존 (repository._load()에서 복원)
#
#   · rule_like        → 규칙이 아닌 컨텍스트 텍스트. payload rules에서 제외
#                        (build_payload 단계에서 필터링)
#   · rule_like_*      → DB 타입은 allowed/disallowed/limit 유지,
#                        metadata.source_kind="heuristic", confidence="low"
#   · category         → DB 타입 "category" 유지 (progress로 바꾸지 않음)
#   · qa_allowed       → DB 타입 "allowed", metadata.source_kind="qa"
#   · qa_disallowed    → DB 타입 "disallowed", metadata.source_kind="qa"
#   · qa_limit         → DB 타입 "limit", metadata.source_kind="qa"
# --------------------------------------------------------------------------
_V2_RULE_TYPE: dict[str, str] = {
    # rule_like 계열: 의미 있는 타입으로 저장, 출처는 metadata로
    "rule_like_allowed": "allowed",
    "rule_like_disallowed": "disallowed",
    "rule_like_limit": "limit",
    # qa 계열: 각 의미별 타입으로 저장 (단일 "qa"로 뭉개지 않음)
    "qa_allowed": "allowed",
    "qa_disallowed": "disallowed",
    "qa_limit": "limit",
    # category는 그대로 유지 (progress로 변환 금지)
    # rule_like(순수 컨텍스트)는 build_payload에서 rules에 포함하지 않음
}
# source_kind: DB 타입은 같아도 원본 출처를 metadata로 보존
_SOURCE_KIND_BY_RULE_TYPE: dict[str, str] = {
    "rule_like_allowed": "heuristic",
    "rule_like_disallowed": "heuristic",
    "rule_like_limit": "heuristic",
    "qa_allowed": "qa",
    "qa_disallowed": "qa",
    "qa_limit": "qa",
}
_CONFIDENCE_BY_RULE_TYPE: dict[str, str] = {
    "rule_like_allowed": "low",
    "rule_like_disallowed": "low",
    "rule_like_limit": "low",
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


def _fix_ocr_split(text: str) -> str:
    """OCR로 인해 한 글자씩 띄어진 한글을 복원한다. 예: '사 역' → '사역'"""
    # 한 글자 + 공백 + 한 글자 패턴이 연속되는 경우만 붙임 (오탐 방지)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(?<=[가-힣]) (?=[가-힣])(?=.{0,60}(?:\s|$))", "", text, count=1)
        # 과도한 제거 방지: 일반 단어 사이 공백은 유지하므로 패턴 재확인
        break
    # 안전한 방식: 연속 3글자 이상 한 글자씩 띄어진 패턴만 붙이기
    def _join_split(m: re.Match) -> str:
        return m.group(1) + m.group(2) + m.group(3)
    return _OCR_SPLIT_RE.sub(_join_split, text)


def _extract_qa_keyword(question: str) -> str:
    """QA 질문 전체에서 핵심 항목명만 추출.

    '전산볼트용 추락방지대 구입비를 안전시설비 항목으로 사용 가능한지 전기업체...'
    → '전산볼트용 추락방지대 구입비를 안전시설비 항목으로'

    keyword가 너무 길면 validator 토큰 매칭에서 false positive가 발생하므로
    '사용 가능한지' 앞부분만 잘라내고 80자로 제한한다.
    """
    q = _normalize_whitespace(question or "")
    for marker in ("사용 가능한지", "사용이 가능한지", "가능한지"):
        if marker in q:
            q = q.split(marker)[0].strip()
            break
    # 후치 조사 제거
    q = re.sub(r"[을를이가은는]$", "", q).strip()
    return q[:80] if len(q) > 80 else q


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


def _parse_rate_to_float(text: str) -> float | None:
    """'3.11%' → 0.0311  /  '3.11' → 0.0311"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?", text.strip())
    if not m:
        return None
    return round(float(m.group(1)) / 100, 6)


def _parse_amount_to_int(text: str) -> int | None:
    """'4,325,000원' → 4325000"""
    m = re.search(r"([\d,]+)\s*원", text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


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
    if _NOISE_RE.match(stripped):
        return True
    # TOC형: 점선 목차 패턴 (예: "| 01 해설집···· · · · ·")
    if _TOC_LINE_RE.search(stripped):
        return True
    # 영문 단독 쓰레기 행 (30자 미만 영문만 있는 줄)
    if len(stripped) < 30 and _ENGLISH_JUNK_RE.match(stripped):
        return True
    return False


def _is_toc_block(body: str) -> bool:
    """본문 전체가 목차/TOC 블록인지 판단."""
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    if not lines:
        return False
    toc_count = sum(1 for l in lines if _TOC_LINE_RE.search(l))
    return toc_count >= max(2, len(lines) * 0.5)


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

        # OCR 띄어짐 보정 적용
        cleaned_lines = [_fix_ocr_split(line) for line in cleaned_lines]

        body = "\n".join(cleaned_lines).strip()
        article_match = _ARTICLE_HEADER_RE.match(cleaned_lines[0].lstrip("- ").strip())
        article_no = f"제{article_match.group(1)}조" if article_match else None

        # 30자 미만 짧은 청크는 의미없는 노이즈로 제거 (실제 조문 제외)
        if len(body) < 30 and not article_no:
            buffer_lines = []
            buffer_cites = []
            return

        # TOC 블록 전체 제거 (목차 페이지는 규칙 검색에 도움이 안 됨)
        if _is_toc_block(body):
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


# 부록 분해용 패턴
# 가. 나. 다. ... 대항목
_DISALLOWED_SECTION_RE = re.compile(r"^([가-하])\.\s*(.+)")
# 1) 2) 3) 세부항목
_DISALLOWED_ITEM_RE = re.compile(r"^(\d+)\)\s*(.+)")
# 가) 나) 세세항목
_DISALLOWED_SUBITEM_RE = re.compile(r"^([가-하])\)\s*(.+)")


def _split_disallowed_body(body: str) -> list[tuple[str, str, list[str]]]:
    """
    사용불가 본문을 가./나. 대항목 단위로 분해한다.
    반환: [(section_label, section_text, sub_keywords), ...]
    예: [("가", "원활한 공사수행을 위한 가설시설...", ["가설울타리", "비계", ...]), ...]
    """
    # 공백 정규화
    body = _normalize_whitespace(body)

    # 가. 나. 다. 단위로 분리
    section_pattern = re.compile(r"(?<!\w)([가-하])\.\s+")
    parts = section_pattern.split(body)

    # parts: [intro_text, label, content, label, content, ...]
    sections: list[tuple[str, str, list[str]]] = []

    # 첫 도입부 (가. 이전 텍스트)
    intro = parts[0].strip() if parts else ""

    i = 1
    while i + 1 < len(parts):
        label = parts[i]
        content = parts[i + 1].strip()
        if not content:
            i += 2
            continue

        # 세부 키워드 추출 (1) 2) 3) 항목들) — inline 텍스트이므로 전방탐색 패턴 사용
        sub_keywords: list[str] = []
        # "1) 텍스트 2) 텍스트" 형태에서 각 항목 추출
        item_inline_re = re.compile(r"\d+\)\s*([^0-9※\*\n]+?)(?=\s*\d+\)|$|[※\*])")
        for item_match in item_inline_re.finditer(content):
            kw = _normalize_whitespace(item_match.group(1))
            kw = re.split(r"[※\*]", kw)[0].strip()
            kw = kw.rstrip("등 ")
            if kw and 4 <= len(kw) <= 60:
                sub_keywords.append(kw)
            elif kw and len(kw) > 60:
                sub_keywords.append(kw[:50])

        # 섹션 텍스트 = 도입부 + 현재 항목
        section_text = f"{label}. {content}"
        if intro:
            section_text = f"{intro} {section_text}"

        sections.append((label, section_text, sub_keywords))
        i += 2

    # 가. 항목이 없으면 본문 전체를 하나로
    if not sections and body:
        sections = [("전체", body, [])]

    return sections


def parse_disallowed_rules(final_md: Path, source_id: str) -> list[LegalRule]:
    rules: list[LegalRule] = []

    for line in final_md.read_text(encoding="utf-8").splitlines():
        match = _APPENDIX_ROW_RE.match(line.strip())
        if not match:
            continue

        category_number = int(match.group(2))
        category_code = _CATEGORY_CODES.get(category_number)
        category_name = _CATEGORY_NAMES.get(category_number, _normalize_whitespace(match.group(3)))
        legal_basis = _normalize_whitespace(match.group(1))
        raw_body = _normalize_whitespace(match.group(4))

        if raw_body == "-":
            # 내용 없는 항목 (예: CAT_07)
            rules.append(
                LegalRule(
                    rule_id=f"{source_id}:disallowed:{category_number}",
                    source_id=source_id,
                    rule_type="disallowed",
                    category_code=category_code,
                    category_number=category_number,
                    category_name=category_name,
                    allowed=False,
                    keyword=None,
                    item_pattern=None,
                    legal_basis=legal_basis,
                    limit_pct=None,
                    rule_text="-",
                    metadata={
                        "raw_category_name": _normalize_whitespace(match.group(3)),
                        "note_only": True,
                        "note": "원문 표에서 사용불가 내역이 '-'로 표기됨",
                    },
                )
            )
            continue

        # 가./나. 단위로 분해
        sections = _split_disallowed_body(raw_body)

        if len(sections) <= 1:
            # 분해 불가 → 기존 방식으로 단일 row
            rules.append(
                LegalRule(
                    rule_id=f"{source_id}:disallowed:{category_number}",
                    source_id=source_id,
                    rule_type="disallowed",
                    category_code=category_code,
                    category_number=category_number,
                    category_name=category_name,
                    allowed=False,
                    keyword=None,
                    item_pattern=None,
                    legal_basis=legal_basis,
                    limit_pct=None,
                    rule_text=_clean_rule_text_for_storage(raw_body, rule_type="disallowed"),
                    metadata={"raw_category_name": _normalize_whitespace(match.group(3))},
                )
            )
        else:
            for sec_idx, (label, section_text, sub_keywords) in enumerate(sections, start=1):
                cleaned_text = _clean_rule_text_for_storage(section_text, rule_type="disallowed")
                # keyword: 첫 번째 세부항목 or 섹션 첫 줄 요약
                keyword = sub_keywords[0] if sub_keywords else cleaned_text[:40]
                # item_pattern: 세부 키워드 전체 (매칭용)
                item_pattern = " | ".join(sub_keywords) if sub_keywords else None

                rules.append(
                    LegalRule(
                        rule_id=f"{source_id}:disallowed:{category_number}:{label}",
                        source_id=source_id,
                        rule_type="disallowed",
                        category_code=category_code,
                        category_number=category_number,
                        category_name=category_name,
                        allowed=False,
                        keyword=keyword,
                        item_pattern=item_pattern,
                        legal_basis=legal_basis,
                        limit_pct=None,
                        rule_text=cleaned_text,
                        metadata={
                            "raw_category_name": _normalize_whitespace(match.group(3)),
                            "section_label": label,
                            "section_index": sec_idx,
                            "sub_keywords": sub_keywords,
                        },
                    )
                )

    # CAT_07이 없으면 추가
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
                metadata={"note": "원문 표에서 사용불가 내역이 '-'로 표기됨", "note_only": True},
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

        # [6] 숫자/한도 구조화 — string values → proper numeric types
        under_5b_rate: float | None = _parse_rate_to_float(under_5)
        between_rate: float | None = _parse_rate_to_float(between_5_50)
        base_amount_int: int | None = _parse_amount_to_int(base_amount)
        over_50b_rate: float | None = _parse_rate_to_float(over_50)
        manager_rate_float: float | None = _parse_rate_to_float(manager_rate)

        # human-readable rule_text still uses original string labels
        rule_text = (
            f"{name}의 산업안전보건관리비 계상기준은 대상액 5억 원 미만 {under_5}, "
            f"5억 원 이상 50억 원 미만 적용비율 {between_5_50}와 기초액 {base_amount}, "
            f"50억 원 이상 {over_50}, 보건관리자 선임 대상 공사 {manager_rate}이다."
        )
        rules.append(
            _make_rule(
                rule_id=f"{source_id}:appendix1:{idx}",
                source_id=source_id,
                rule_type="limit",
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
                    # structured numeric fields
                    "under_5b_rate_pct": under_5b_rate,
                    "between_5b_50b_rate_pct": between_rate,
                    "between_5b_50b_base_amount": base_amount_int,
                    "over_50b_rate_pct": over_50b_rate,
                    "manager_target_rate_pct": manager_rate_float,
                    # raw strings for traceability
                    "raw_under_5b": under_5,
                    "raw_between_5b_50b": between_5_50,
                    "raw_base_amount": base_amount,
                    "raw_over_50b": over_50,
                    "raw_manager_rate": manager_rate,
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


_QC_THRESHOLDS = {
    "null_ratio": 0.10,       # > 10% empty rule_text → warning
    "note_only_ratio": 0.20,  # > 20% note_only rules → warning
    "heuristic_ratio": 0.40,  # > 40% rule_like/heuristic rules → warning
    "min_rules_per_source": 1,
}

_HEURISTIC_SOURCE_KINDS = {"heuristic"}


def _rule_qc_metrics(rules: list) -> dict:
    """Compute quality metrics for a list of LegalRule objects."""
    total = len(rules)
    if total == 0:
        return {"total": 0, "warnings": ["no rules found"]}

    null_count = sum(
        1 for r in rules
        if not r.rule_text or r.rule_text.strip() in ("-", "")
    )
    note_only_count = sum(
        1 for r in rules
        if (r.metadata or {}).get("note_only") is True
    )
    heuristic_count = sum(
        1 for r in rules
        if (r.metadata or {}).get("source_kind") in _HEURISTIC_SOURCE_KINDS
    )
    no_category_count = sum(
        1 for r in rules
        if not r.category_code and r.rule_type not in ("progress", "limit")
    )

    null_ratio = null_count / total
    note_only_ratio = note_only_count / total
    heuristic_ratio = heuristic_count / total

    warnings: list[str] = []
    if null_ratio > _QC_THRESHOLDS["null_ratio"]:
        warnings.append(f"null_ratio {null_ratio:.1%} > threshold {_QC_THRESHOLDS['null_ratio']:.0%}")
    if note_only_ratio > _QC_THRESHOLDS["note_only_ratio"]:
        warnings.append(f"note_only_ratio {note_only_ratio:.1%} > threshold {_QC_THRESHOLDS['note_only_ratio']:.0%}")
    if heuristic_ratio > _QC_THRESHOLDS["heuristic_ratio"]:
        warnings.append(f"heuristic_ratio {heuristic_ratio:.1%} > threshold {_QC_THRESHOLDS['heuristic_ratio']:.0%}")

    rule_type_dist: dict[str, int] = {}
    for r in rules:
        rule_type_dist[r.rule_type] = rule_type_dist.get(r.rule_type, 0) + 1

    return {
        "total": total,
        "null_count": null_count,
        "null_ratio": round(null_ratio, 4),
        "note_only_count": note_only_count,
        "note_only_ratio": round(note_only_ratio, 4),
        "heuristic_count": heuristic_count,
        "heuristic_ratio": round(heuristic_ratio, 4),
        "no_category_count": no_category_count,
        "rule_type_distribution": rule_type_dist,
        "qc_passed": len(warnings) == 0,
        "warnings": warnings,
    }


def build_verification_report(
    documents: list[SourceDocument],
    corpus_entries: list[CorpusEntry],
    rule_entries: list | None = None,
) -> dict:
    """Build a structured quality report covering corpus coverage and rule quality."""
    rule_entries = rule_entries or []
    rules_by_source: dict[str, list] = {}
    for r in rule_entries:
        rules_by_source.setdefault(r.source_id, []).append(r)

    report: dict[str, dict] = {}
    for source in documents:
        source_text = Path(source.source_path).read_text(encoding="utf-8")
        source_lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip() and not line.strip().startswith("<!--")
        ]
        src_entries = [e for e in corpus_entries if e.source_id == source.source_id]
        covered_entries = len(src_entries)
        cited_count = sum(len(e.cited_laws) for e in src_entries)
        src_rules = rules_by_source.get(source.source_id, [])

        report[source.source_id] = {
            "source_name": source.source_name,
            "source_type": source.source_type,
            "nonempty_source_lines": len(source_lines),
            "corpus_entries": covered_entries,
            "cited_laws_found": cited_count,
            "has_corpus_entries": covered_entries > 0,
            "rule_qc": _rule_qc_metrics(src_rules),
        }

    # global QC rollup
    all_warnings = []
    for sid, data in report.items():
        for w in data["rule_qc"].get("warnings", []):
            all_warnings.append(f"[{sid}] {w}")

    report["__global__"] = {
        "total_sources": len(documents),
        "total_corpus_entries": len(corpus_entries),
        "total_rules": len(rule_entries),
        "global_qc": _rule_qc_metrics(rule_entries),
        "all_warnings": all_warnings,
        "qc_passed": len(all_warnings) == 0,
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
                keyword=_extract_qa_keyword(current_question),
                item_pattern=_extract_qa_keyword(current_question),
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
    # rule_like(순수 컨텍스트 텍스트)는 규칙이 아니므로 payload rules에서 제외
    # rule_like_allowed/disallowed/limit 계열은 유지 (실제 판정에 사용 가능)
    rule_entries = [r for r in rule_entries if r.rule_type != "rule_like"]
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


def payload_to_sql(payload: dict, *, full_refresh: bool = False) -> str:
    """Generate SQL to load the payload into the legal_db.

    Args:
        payload: the dict produced by build_payload().
        full_refresh: if True, TRUNCATE all tables before inserting (replaces everything).
                      if False (default), DELETE rows for only the source_ids present in
                      the payload, then INSERT — safe for incremental source updates.
    """
    statements = ["BEGIN;", "SET LOCAL search_path TO legal_rag, public;"]

    source_ids = list({row["source_id"] for row in payload.get("documents", [])})
    sid_list = ", ".join(_sql_literal(sid) for sid in source_ids)

    if full_refresh:
        statements.append(
            "TRUNCATE legal_rule_master, legal_rule_profiles, legal_citations, legal_rules, legal_corpus, legal_source_documents RESTART IDENTITY CASCADE;"
        )
    else:
        # [8] source-scoped DELETE so other sources are untouched
        if sid_list:
            statements.extend([
                f"DELETE FROM legal_rule_master WHERE source_id IN ({sid_list});",
                f"DELETE FROM legal_citations   WHERE source_id IN ({sid_list});",
                f"DELETE FROM legal_rules        WHERE source_id IN ({sid_list});",
                f"DELETE FROM legal_corpus       WHERE source_id IN ({sid_list});",
                f"DELETE FROM legal_source_documents WHERE source_id IN ({sid_list});",
            ])
        # rule_profiles are config (not per-source) — always replace entirely
        statements.append("DELETE FROM legal_rule_profiles WHERE true;")

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
        metadata = {**row["metadata"]}
        # 원본 타입 보존 (복원 가능하도록)
        if v2_rt != original_rt:
            metadata["original_rule_type"] = original_rt
        # 출처(source_kind)와 신뢰도(confidence) 기록
        if original_rt in _SOURCE_KIND_BY_RULE_TYPE:
            metadata.setdefault("source_kind", _SOURCE_KIND_BY_RULE_TYPE[original_rt])
        if original_rt in _CONFIDENCE_BY_RULE_TYPE:
            metadata.setdefault("confidence", _CONFIDENCE_BY_RULE_TYPE[original_rt])
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
    parser.add_argument("--full-refresh", action="store_true", help="전체 테이블 TRUNCATE 후 재적재 (기존 데이터 완전 교체). 기본은 소스 단위 DELETE+INSERT.")
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
    sql_out.write_text(payload_to_sql(payload, full_refresh=args.full_refresh), encoding="utf-8")

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
