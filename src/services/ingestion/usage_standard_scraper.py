# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-22
# 수정일   : 2026-05-22
#
# [ 주요 함수 정의 ]
#
# 1. fetch_usage_standard()        : 건설업 산안비 계상 및 사용기준 API/HTML 수집
# 2. convert_to_markdown()         : 원문 텍스트 → 조/항/호/목 계층 마크다운 변환
# 3. upsert_to_rdb()               : legal_master(corpus+rule) + legal_rule_profiles upsert
# 4. run_usage_standard_pipeline() : 전체 파이프라인
# --------------------------------------------------------------------------
"""
건설업 산업안전보건관리비 계상 및 사용기준 (고용노동부 고시)
https://www.law.go.kr/LSW/admRulInfoR.do?admRulSeq=2100000254546

고시 전문을 조/항/호/목 계층으로 파싱 후:
  - Qdrant        : legal_master rows(조·별표·rule 단위)를 Document로 변환, chunk_id=uuid5(master_id) 고정
  - legal_master  : corpus(조문·별표) + rule(제7조 각호 allowed / 별표2 disallowed) 통합 적재
  - legal_rule_profiles : seed_legal_rule_profiles.json 기반 운영형 보강 룰

chunk_id 가 Qdrant point ID 와 1:1 대응되므로, 새벽 배치에서 hash 변경 감지 후
해당 chunk_id 만 삭제·재임베딩하는 청크 단위 refresh 가 가능하다.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

from src.core.storage import DEFAULT_COLLECTION, make_chunk_id, upsert_with_ids

log = logging.getLogger(__name__)

_DRF_BASE_URL = "https://www.law.go.kr/DRF"
_ADMRUL_NAME = "건설업 산업안전보건관리비 계상 및 사용기준"
_USAGE_STANDARD_URL = (
    "https://www.law.go.kr/LSW/admRulInfoR.do?admRulSeq=2100000254546"
)
_HTML_CACHE_PATH = Path(".cache/usage_standard_raw.html")
_HTML_CACHE_TTL  = 60 * 60 * 24  # 24시간 (초 단위)
_SOURCE_ID    = "usage_standard:admRulSeq:2100000254546"
_SOURCE_NAME  = "건설업 산업안전보건관리비 계상 및 사용기준(고용노동부고시)(제2025-11호)(20250212)"
_SOURCE_TITLE = "건설업 산업안전보건관리비 계상 및 사용기준"
_EFFECTIVE_DATE = "2025-02-12"
_NOTICE_NO    = "제2025-11호"
_LAW_PREFIX   = "건설업 산업안전보건관리비 계상 및 사용기준"

_HO_TO_CAT: dict[int, str] = {
    1: "CAT_01", 2: "CAT_02", 3: "CAT_03",
    4: "CAT_04", 5: "CAT_05", 6: "CAT_06",
    7: "CAT_07", 8: "CAT_08", 9: "CAT_09",
}
_CAT_NAMES: dict[str, str] = {
    "CAT_01": "안전관리자 등의 인건비 및 각종 업무 수당 등",
    "CAT_02": "안전시설비 등",
    "CAT_03": "보호구 등",
    "CAT_04": "안전보건진단비 등",
    "CAT_05": "안전보건교육비 등",
    "CAT_06": "근로자 건강장해예방비 등",
    "CAT_07": "건설재해예방 기술지도비",
    "CAT_08": "본사 안전전담부서 운영비",
    "CAT_09": "위험성평가 등에 따른 소요비용",
}


# ── 법제처 행정규칙 Open API 수집 ───────────────────────────────────────────

def _api_key() -> str:
    """LAW_API_KEY를 호출 시점에 읽는다."""
    return os.environ.get("LAW_API_KEY", "").strip()


def _law_api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    request_params = {
        **params,
        "OC": _api_key(),
        "type": "JSON",
    }
    resp = requests.get(f"{_DRF_BASE_URL}/{endpoint}", params=request_params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("법제처 API 응답이 JSON object가 아닙니다.")
    return data


def _fetch_current_admrul_summary() -> dict[str, Any]:
    if not _api_key():
        raise RuntimeError("LAW_API_KEY 환경변수가 없어 행정규칙 API를 사용할 수 없습니다.")

    data = _law_api_get(
        "lawSearch.do",
        {
            "target": "admrul",
            "mobileYn": "Y",
            "nw": "1",
            "query": _ADMRUL_NAME,
            "sort": "efdes",
        },
    )
    search = data.get("AdmRulSearch") or {}
    admrul = search.get("admrul")
    if isinstance(admrul, list):
        admrul = admrul[0] if admrul else None
    if not isinstance(admrul, dict):
        raise RuntimeError(f"현행 행정규칙 검색 결과가 없습니다: {_ADMRUL_NAME}")

    seq = str(admrul.get("행정규칙일련번호") or "").strip()
    if not seq:
        raise RuntimeError("현행 행정규칙 검색 결과에 행정규칙일련번호가 없습니다.")
    return admrul


def _fetch_admrul_service(admrul_seq: str) -> dict[str, Any]:
    data = _law_api_get(
        "lawService.do",
        {
            "target": "admrul",
            "ID": admrul_seq,
        },
    )
    service = data.get("AdmRulService")
    if not isinstance(service, dict):
        raise RuntimeError(f"행정규칙 본문 응답이 비어 있습니다: ID={admrul_seq}")
    return service


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            lines.extend(_flatten_text(item))
        return lines
    if isinstance(value, dict):
        lines: list[str] = []
        for item in value.values():
            lines.extend(_flatten_text(item))
        return lines
    text = str(value).strip()
    return [text] if text else []


def _format_yyyymmdd(value: Any) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) != 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _display_yyyymmdd(value: Any) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) != 8:
        return None
    return f"{digits[:4]}. {int(digits[4:6])}. {int(digits[6:])}."


def _normalize_admrul_text(lines: list[str]) -> str:
    text = "\n".join(line.strip() for line in lines if str(line).strip())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _admrul_service_to_text(service: dict[str, Any]) -> str:
    info = service.get("행정규칙기본정보") or {}
    if not isinstance(info, dict):
        info = {}

    title = str(info.get("행정규칙명") or _LAW_PREFIX).strip()
    kind = str(info.get("행정규칙종류") or "고시").strip()
    notice_no = str(info.get("발령번호") or "").strip()
    effective = _display_yyyymmdd(info.get("시행일자"))

    lines: list[str] = [title]
    if effective:
        lines.append(f"[시행 {effective}]")
    if notice_no:
        lines.append(f"[고용노동부{kind} 제{notice_no}호]")
    lines.append("")

    lines.extend(_flatten_text(service.get("조문내용")))

    appendix = service.get("별표") or {}
    appendix_units = appendix.get("별표단위") if isinstance(appendix, dict) else []
    if isinstance(appendix_units, dict):
        appendix_units = [appendix_units]
    if isinstance(appendix_units, list):
        for unit in appendix_units:
            if not isinstance(unit, dict):
                continue
            body_lines = _flatten_text(unit.get("별표내용"))
            if not body_lines:
                continue
            lines.append("")
            lines.extend(body_lines)

    bylaws = service.get("부칙") or {}
    bylaw_lines = _flatten_text(bylaws.get("부칙내용") if isinstance(bylaws, dict) else bylaws)
    if bylaw_lines:
        lines.append("")
        lines.extend(bylaw_lines)

    return _normalize_admrul_text(lines)


def _update_source_metadata(summary: dict[str, Any], service: dict[str, Any]) -> None:
    global _SOURCE_ID, _SOURCE_NAME, _SOURCE_TITLE, _EFFECTIVE_DATE, _NOTICE_NO
    global _USAGE_STANDARD_URL, _LAW_PREFIX

    info = service.get("행정규칙기본정보") or {}
    if not isinstance(info, dict):
        info = {}

    admrul_seq = str(
        info.get("행정규칙일련번호")
        or summary.get("행정규칙일련번호")
        or "2100000254546"
    ).strip()
    title = str(info.get("행정규칙명") or summary.get("행정규칙명") or _ADMRUL_NAME).strip()
    kind = str(info.get("행정규칙종류") or summary.get("행정규칙종류") or "고시").strip()
    notice_no = str(info.get("발령번호") or summary.get("발령번호") or "").strip()
    effective_raw = info.get("시행일자") or summary.get("시행일자")
    effective_date = _format_yyyymmdd(effective_raw)
    effective_compact = re.sub(r"\D", "", str(effective_raw or ""))

    _SOURCE_ID = f"usage_standard:admRulSeq:{admrul_seq}"
    _SOURCE_TITLE = title
    _LAW_PREFIX = title
    _NOTICE_NO = f"제{notice_no}호" if notice_no else ""
    _EFFECTIVE_DATE = effective_date or ""
    _SOURCE_NAME = (
        f"{title}(고용노동부{kind})"
        f"({_NOTICE_NO})"
        f"({effective_compact})"
    )
    _USAGE_STANDARD_URL = (
        "https://www.law.go.kr/DRF/lawService.do"
        f"?target=admrul&ID={admrul_seq}&type=JSON"
    )


def _fetch_usage_standard_from_api() -> str:
    summary = _fetch_current_admrul_summary()
    admrul_seq = str(summary["행정규칙일련번호"])
    service = _fetch_admrul_service(admrul_seq)
    _update_source_metadata(summary, service)
    text = _admrul_service_to_text(service)
    if len(text) < 1000:
        raise RuntimeError(f"행정규칙 API 원문 길이가 너무 짧습니다: {len(text)}자")
    log.info(
        "행정규칙 API 수집 완료: seq=%s, source=%s, text=%d자",
        admrul_seq,
        _SOURCE_NAME,
        len(text),
    )
    return text


# ── HTML 수집 ────────────────────────────────────────────────────────────────

def _fetch_html(force_refresh: bool = False) -> str:
    """
    law.go.kr HTML을 가져오되, 24시간 이내 캐시가 있으면 재사용한다.

    law.go.kr는 세션 없이 직접 접근하면 차단(SSL EOF / 404)하므로:
      1. requests.Session으로 메인 페이지를 먼저 방문해 쿠키를 획득
      2. Referer 헤더를 포함해 본문 URL 요청

    성공한 HTML을 .cache/usage_standard_raw.html 에 저장하여
    이후 실행에서 재수집 없이 재사용한다.
    """
    import time

    if not force_refresh and _HTML_CACHE_PATH.exists():
        age = time.time() - _HTML_CACHE_PATH.stat().st_mtime
        if age < _HTML_CACHE_TTL:
            log.info("캐시 사용 (%.0f시간 전 수집): %s", age / 3600, _HTML_CACHE_PATH)
            return _HTML_CACHE_PATH.read_text(encoding="utf-8")
        log.info("캐시 만료 (%.0f시간 경과) — 재수집", age / 3600)

    _BASE_URL = "https://www.law.go.kr"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    }

    session = requests.Session()
    session.headers.update(headers)

    # 1단계: 메인 페이지 방문 → 쿠키 획득
    log.info("law.go.kr 메인 페이지 방문 (쿠키 획득)")
    session.get(_BASE_URL, timeout=20)

    # 2단계: Referer 포함 본문 요청
    log.info("고시 본문 수집: %s", _USAGE_STANDARD_URL)
    session.headers["Referer"] = _BASE_URL
    resp = session.get(_USAGE_STANDARD_URL, timeout=30)

    if resp.status_code != 200 or len(resp.text) < 1000:
        raise RuntimeError(
            f"law.go.kr 응답 이상 — status={resp.status_code}, "
            f"본문길이={len(resp.text)}자. "
            "브라우저로 직접 접속해 HTML을 .cache/usage_standard_raw.html 에 저장하세요."
        )

    _HTML_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HTML_CACHE_PATH.write_text(resp.text, encoding="utf-8")
    log.info("HTML 캐시 저장 완료: %s (%d bytes)", _HTML_CACHE_PATH, len(resp.text))
    return resp.text


def fetch_usage_standard(force_refresh: bool = False) -> str:
    """
    고시 전문을 법제처 행정규칙 API 우선으로 수집해 정제된 텍스트를 반환한다.

    API 수집에 실패하면 기존 law.go.kr HTML 스크래핑을 fallback으로 사용한다.
    force_refresh=True 이면 HTML fallback 시 캐시를 무시하고 재수집한다.

    HTML fallback 핵심 처리:
      1. <a> 태그 unwrap  — 인라인 법령 참조(별표1, 제72조 등)가
         separator="\n" 에 의해 독립 줄로 쪼개지는 것 방지
      2. <script>/<style> 제거
      3. block 요소(\n) vs inline 요소(공백) 구분하여 텍스트 추출
    """
    from bs4 import BeautifulSoup, NavigableString

    try:
        return _fetch_usage_standard_from_api()
    except Exception as e:
        log.warning("행정규칙 API 수집 실패 — HTML fallback 사용: %s", e)

    html = _fetch_html(force_refresh=force_refresh)
    soup = BeautifulSoup(html, "html.parser")

    # 불필요 태그 제거
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # <a> 태그 unwrap: 링크 텍스트를 주변 텍스트에 인라인으로 합침
    # → 별표1, 제72조 등 hyperlink 가 별도 줄로 분리되는 문제 방지
    for a_tag in soup.find_all("a"):
        a_tag.unwrap()

    # 본문 컨테이너 선택 (우선순위 순)
    container = None
    for selector in [
        "#lawViewWrap", "#conWrap", "#lawContWrap",
        "div.law_view", "div.view_wrap", "div.law_cont",
        "article", "main",
    ]:
        container = soup.select_one(selector)
        if container:
            log.info("본문 컨테이너 selector: %s", selector)
            break

    target = container if container else soup

    # block 레벨 태그 앞뒤에 줄바꿈 삽입
    _BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                   "tr", "table", "section", "article", "blockquote"}
    for tag in target.find_all(_BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))

    body_text = target.get_text(separator="")

    # HTML 엔티티 처리
    for ent, ch in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&nbsp;", " ")]:
        body_text = body_text.replace(ent, ch)

    # 줄 내 연속 공백 정리, 3줄 이상 빈 줄 압축
    body_text = re.sub(r"[ \t]+", " ", body_text)
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)
    body_text = "\n".join(line.strip() for line in body_text.splitlines())
    log.info("텍스트 추출 완료: %d자", len(body_text))
    return body_text.strip()


# ── 마크다운 변환 ─────────────────────────────────────────────────────────────

def convert_to_markdown(text: str) -> str:
    """
    원문 텍스트를 조/항/호/목 계층 마크다운으로 변환한다.

    헤더 매핑 (PDF 파이프라인의 _HEADERS_TO_SPLIT과 동일):
      #    → 법령명 (최상위)
      ##   → 제X조 (조문)
      ###  → ①②③ (항)
      #### → 1. 2. 3. (호)
              가. 나. 다. (목) — 호 본문 안에 포함
    """
    lines_out: list[str] = [f"# {_LAW_PREFIX}\n"]

    # fetch_usage_standard()가 줄바꿈을 보존하므로 줄 단위로 처리
    lines = text.splitlines()

    # ── 조문/별표 경계 탐지 ───────────────────────────────────────────────────
    # 실제 별표 섹션 시작: 줄이 [별표N] 또는 "별표N" (조사 없이 단독)으로 시작
    _APPENDIX_LINE_RE = re.compile(
        r"^\s*(?:\[별표\s*(\d+(?:의\d+)?)\]|별표\s*(\d+(?:의\d+)?)\s*$)"
    )
    # 조문 시작: 줄이 "제N조" 로 시작
    _ARTICLE_LINE_RE = re.compile(r"^\s*제(\d+)조(?:의\d+)?\s*[(\s]")

    # 전체 텍스트를 조문/별표 블록으로 재조립
    blocks: list[dict] = []  # {"type": "article"|"appendix", "key": str, "lines": []}
    current: dict | None = None

    for line in lines:
        line_s = line.strip()

        bt_m  = _APPENDIX_LINE_RE.match(line_s)
        art_m = _ARTICLE_LINE_RE.match(line_s)

        if bt_m:
            bt_num = bt_m.group(1) or bt_m.group(2)
            bt_key = f"별표{bt_num}"
            if current:
                blocks.append(current)
            current = {"type": "appendix", "key": bt_key, "lines": []}
        elif art_m:
            if current:
                blocks.append(current)
            current = {"type": "article", "key": f"제{art_m.group(1)}조", "lines": [line]}
        else:
            if current:
                current["lines"].append(line)
            # 조문 시작 전 머리말은 무시

    if current:
        blocks.append(current)

    log.info(
        "블록 탐지: 조문 %d개, 별표 %d개",
        sum(1 for b in blocks if b["type"] == "article"),
        sum(1 for b in blocks if b["type"] == "appendix"),
    )

    # ── 조문 블록 → 마크다운 ─────────────────────────────────────────────────
    for block in blocks:
        block_text = "\n".join(block["lines"]).strip()
        if not block_text:
            continue

        if block["type"] == "article":
            art_m2 = re.match(r"(제\d+조(?:의\d+)?)\s*\(([^)]+)\)\s*(.*)", block_text, re.DOTALL)
            if not art_m2:
                continue
            art_no    = art_m2.group(1)
            art_title = art_m2.group(2)
            art_body  = art_m2.group(3).strip()

            # 조문 전체가 삭제된 경우 스킵
            if _is_deleted(art_body):
                log.debug("삭제 조문 스킵: %s", art_no)
                continue

            lines_out.append(f"\n## {art_no} ({art_title})\n")

            # 항(①②③...) 분리
            para_chunks = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])", art_body)
            for para in para_chunks:
                para = para.strip()
                if not para:
                    continue

                para_m = re.match(r"([①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)", para, re.DOTALL)
                if para_m:
                    para_mark = para_m.group(1)
                    para_body = para_m.group(2).strip()
                    lines_out.append(f"\n### {para_mark}\n")
                else:
                    para_body = para

                # 호(1. 2. ...) 분리 — 숫자. 뒤에 한국어 시작하는 것만
                ho_chunks = re.split(r"(?<!\d)(?=\d+\.\s+[가-힣「『])", para_body)
                for ho_part in ho_chunks:
                    ho_part = ho_part.strip()
                    if not ho_part:
                        continue

                    ho_m = re.match(r"(\d+)\.\s+(.*)", ho_part, re.DOTALL)
                    if ho_m:
                        ho_num  = ho_m.group(1)
                        ho_body = ho_m.group(2).strip()

                        # 호가 삭제된 경우 스킵
                        if _is_deleted(ho_body):
                            log.debug("삭제 호 스킵: %s 제%s호", art_no, ho_num)
                            continue

                        cat_hint = ""
                        if art_no == "제7조":
                            cat = _HO_TO_CAT.get(int(ho_num))
                            if cat:
                                cat_hint = f" [{cat}: {_CAT_NAMES[cat]}]"
                        lines_out.append(f"\n#### {ho_num}.{cat_hint}\n")
                        lines_out.append(_format_mo_items(ho_body))
                    else:
                        # 호 기호 없는 본문 — 삭제 내용이면 스킵
                        if not _is_deleted(ho_part):
                            lines_out.append(ho_part + "\n")

        else:  # appendix
            bt_key   = block["key"]
            bt_title = block["lines"][0].strip()[:80] if block["lines"] else bt_key
            bt_body  = block_text

            # 별표가 삭제된 경우 스킵
            if _is_deleted(bt_body):
                log.debug("삭제 별표 스킵: %s", bt_key)
                continue

            lines_out.append(f"\n## {bt_key} ({bt_title})\n")
            lines_out.append(bt_body + "\n")

    return "".join(lines_out)


def _is_deleted(text: str) -> bool:
    """조문/호/별표가 삭제 처리된 경우 True.

    검사 순서:
      1. 전체(줄 구분 없이) 앞부분 20자를 정제 → "삭제"로 시작하면 True
      2. "삭제" 뒤에 공백 없이 다른 내용이 붙은 경우(예: "삭제제2장")도 포착

    단, "삭제를 요청한..."처럼 '삭제' + 조사가 바로 붙는 정상 텍스트는 False.
    """
    if not text.strip():
        return False

    # 첫 번째 실질 줄
    first = next(
        (l for l in text.strip().splitlines() if l.strip()),
        ""
    ).strip()
    clean = re.sub(r"[<>〈〉《》「」]", "", first).strip()

    # 완전 일치
    if clean == "삭제":
        return True

    # "삭제" 뒤에 공백/섹션 경계가 바로 오는 경우
    # ex) "삭제 제2장...", "삭제제2조..."
    # "삭제를", "삭제된" 같은 정상 텍스트는 여기서 걸리지 않음
    if clean.startswith("삭제"):
        remainder = clean[2:].lstrip()          # "삭제" 이후 문자열
        if not remainder:
            return True
        # 뒤가 장/조/항 구분자이면 삭제된 것
        if re.match(r"^제\d|^\d+장|^부칙", remainder):
            return True

    return False


def _format_mo_items(text: str) -> str:
    """가.나.다. 목 항목을 들여쓰기 리스트로 변환."""
    # 가. 나. 다. ... 패턴
    parts = re.split(r"(?<=[^가-힣])(?=[가나다라마바사아자차카타파하]\.)", text)
    if len(parts) <= 1:
        return text + "\n"

    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mo_m = re.match(r"([가나다라마바사아자차카타파하])\.\s*(.*)", part, re.DOTALL)
        if mo_m:
            out.append(f"- {mo_m.group(1)}. {mo_m.group(2).strip()}")
        else:
            out.append(part)
    return "\n".join(out) + "\n"


# ── RDB upsert ───────────────────────────────────────────────────────────────

def upsert_to_rdb(
    raw_text: str,
    database_url: str | None = None,
    seed_path: Path | None = None,
    _rows: list | None = None,
    _profiles: list | None = None,
) -> dict[str, int]:
    """
    1) legal_master  upsert  — corpus(조문·별표) + rule(제7조·별표2) 통합
    2) legal_rule_profiles upsert — seed_legal_rule_profiles.json 기반

    변경 감지: source_name 단위 DELETE → INSERT. hash + chunk_id 컬럼으로 포인트 단위 추적.

    _rows, _profiles: 이미 빌드된 rows를 외부에서 전달 (재빌드 방지).
                      None이면 raw_text 로부터 직접 빌드한다.
    """
    import psycopg

    db_url     = database_url or os.environ.get(
        "DATABASE_URL",
        "postgresql://safety_user:safety_password@localhost:5432/safety",
    )
    _seed_path = seed_path or _DEFAULT_SEED_PATH

    master_rows  = _dedupe_rows_by_id(
        _rows if _rows is not None else (_build_corpus_rows(raw_text) + _build_rules_rows(raw_text))
    )
    profile_rows = _profiles if _profiles is not None else _build_profile_rows(_seed_path)

    conn          = psycopg.connect(db_url)
    master_count  = 0
    profile_count = 0

    with conn:
        with conn.cursor() as cur:

            # ── 1) legal_master ────────────────────────────────────────────
            cur.execute(
                "DELETE FROM legal_rag.legal_master WHERE source_name = %s",
                (_SOURCE_NAME,),
            )
            for row in master_rows:
                cur.execute(
                    """
                    INSERT INTO legal_rag.legal_master
                      (id, source_name, source_type, source_path,
                       article_no, paragraph_no, item_no, section_path,
                       chunk_id, body, record_type, content_type,
                       rule_type, category_code, category_name,
                       allowed, limit_pct, keyword, item_pattern, legal_basis,
                       cited_laws, keywords, hash, metadata)
                    VALUES
                      (%(id)s, %(source_name)s, %(source_type)s, %(source_path)s,
                       %(article_no)s, %(paragraph_no)s, %(item_no)s, %(section_path)s,
                       %(chunk_id)s, %(body)s, %(record_type)s, %(content_type)s,
                       %(rule_type)s, %(category_code)s, %(category_name)s,
                       %(allowed)s, %(limit_pct)s, %(keyword)s, %(item_pattern)s, %(legal_basis)s,
                       %(cited_laws)s, %(keywords)s, %(hash)s, %(metadata)s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                      body         = EXCLUDED.body,
                      hash         = EXCLUDED.hash,
                      keyword      = EXCLUDED.keyword,
                      allowed      = EXCLUDED.allowed,
                      cited_laws   = EXCLUDED.cited_laws,
                      metadata     = EXCLUDED.metadata
                    """,
                    row,
                )
                master_count += 1
            log.info("legal_master upsert 완료: %d개", master_count)

            # ── 2) legal_rule_profiles ─────────────────────────────────────
            cur.execute("DELETE FROM legal_rag.legal_rule_profiles WHERE true")
            for row in profile_rows:
                cur.execute(
                    """
                    INSERT INTO legal_rag.legal_rule_profiles
                      (profile_id, profile_scope, category_code,
                       profile_key, values_json, metadata)
                    VALUES
                      (%(profile_id)s, %(profile_scope)s, %(category_code)s,
                       %(profile_key)s, %(values_json)s::jsonb, %(metadata)s::jsonb)
                    """,
                    row,
                )
                profile_count += 1
            log.info("legal_rule_profiles upsert 완료: %d개", profile_count)

    conn.close()

    corpus_count = sum(1 for r in master_rows if r["record_type"] == "corpus")
    rules_count  = sum(1 for r in master_rows if r["record_type"] == "rule")
    return {
        "master":   master_count,
        "corpus":   corpus_count,
        "rules":    rules_count,
        "profiles": profile_count,
    }


def _dedupe_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last generated row for each stable legal_master id."""
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["id"]] = row
    return list(deduped.values())


# ── corpus rows 구성 (조 단위) → legal_master format ──────────────────────────

def _build_corpus_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict] = []
    idx = 0

    # 조문
    article_chunks = re.split(r"(?=제\d+조(?:의\d+)?\s*\()", text)
    for chunk in article_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        art_m = re.match(r"(제\d+조(?:의\d+)?)\s*\(([^)]+)\)\s*(.*)", chunk, re.DOTALL)
        if not art_m:
            continue

        art_no = art_m.group(1)
        title  = art_m.group(2).strip()
        body   = art_m.group(3).strip()
        if not body or _is_deleted(body):
            continue

        _row_id = f"{_SOURCE_ID}:{art_no}:{idx:04d}"
        rows.append({
            "id":           _row_id,
            "source_name":  _SOURCE_NAME,
            "source_type":  "law",
            "source_path":  _USAGE_STANDARD_URL,
            "article_no":   art_no,
            "paragraph_no": None,
            "item_no":      None,
            "section_path": f"{_LAW_PREFIX} > {art_no} ({title})",
            "chunk_id":     make_chunk_id(_row_id),
            "body":         body,
            "record_type":  "corpus",
            "content_type": "article",
            "rule_type":    None,
            "category_code": None,
            "category_name": None,
            "allowed":      None,
            "limit_pct":    None,
            "keyword":      None,
            "item_pattern": None,
            "legal_basis":  None,
            "cited_laws":   _extract_cited_laws(body),
            "keywords":     [],
            "hash":         hashlib.sha256(body.encode()).hexdigest(),
            "metadata":     json.dumps({"source_kind": "usage_standard", "title": title}, ensure_ascii=False),
        })
        idx += 1

    # 별표
    bt_splits = re.split(r"(\[별표\s*\d+(?:의\d+)?\]|별표\s*\d+(?:의\d+)?(?=\s))", text)
    seen: set[str] = set()
    i = 1
    while i < len(bt_splits) - 1:
        header_raw = bt_splits[i].strip()
        body_raw   = bt_splits[i + 1].strip() if i + 1 < len(bt_splits) else ""
        bt_key     = re.sub(r"\s+", "", re.sub(r"[\[\]]", "", header_raw))

        if bt_key not in seen and len(body_raw) >= 20:
            seen.add(bt_key)
            first_line = body_raw.split("\n")[0][:80].strip()
            _bt_row_id = f"{_SOURCE_ID}:{bt_key}:{idx:04d}"
            rows.append({
                "id":           _bt_row_id,
                "source_name":  _SOURCE_NAME,
                "source_type":  "law",
                "source_path":  _USAGE_STANDARD_URL,
                "article_no":   None,
                "paragraph_no": None,
                "item_no":      None,
                "section_path": f"{_LAW_PREFIX} > {bt_key}",
                "chunk_id":     make_chunk_id(_bt_row_id),
                "body":         body_raw,
                "record_type":  "corpus",
                "content_type": "guideline",
                "rule_type":    None,
                "category_code": None,
                "category_name": None,
                "allowed":      None,
                "limit_pct":    None,
                "keyword":      None,
                "item_pattern": None,
                "legal_basis":  None,
                "cited_laws":   _extract_cited_laws(body_raw),
                "keywords":     [],
                "hash":         hashlib.sha256(body_raw.encode()).hexdigest(),
                "metadata":     json.dumps(
                    {"source_kind": "usage_standard", "section_key": bt_key, "title": first_line or bt_key},
                    ensure_ascii=False,
                ),
            })
            idx += 1
        i += 2

    return rows


# ── rules rows 구성 (제7조 각호 + 별표2) ─────────────────────────────────────

def _build_rules_rows(text: str) -> list[dict[str, Any]]:
    import json
    rows: list[dict] = []

    # 제7조 파싱
    art7_m = re.search(r"제7조\s*\(사용기준\)\s*(.*?)(?=제8조|$)", text, re.DOTALL)
    if art7_m:
        art7_body = art7_m.group(1)
        ho_splits = re.split(r"(?<!\d)(\d)\.\s+", art7_body)
        i = 1
        while i < len(ho_splits) - 1:
            try:
                ho_num = int(ho_splits[i])
            except ValueError:
                i += 1
                continue
            if ho_num not in _HO_TO_CAT:
                i += 2
                continue

            ho_body = ho_splits[i + 1] if i + 1 < len(ho_splits) else ""
            title_m = re.match(r"^([^가나다라마바사아자차카타파하]+?)(?=\s*가\.)", ho_body)
            ho_title = title_m.group(1).strip() if title_m else ho_body[:50].strip()

            sub_items = re.split(r"\s+(?=[가나다라마바사아자차카타파하]\.)", ho_body)
            sub_items = [s.strip() for s in sub_items if s.strip() and len(s.strip()) > 5]

            cat   = _HO_TO_CAT[ho_num]
            basis = f"{_LAW_PREFIX} 제7조제1항제{ho_num}호"

            for sub_idx, item_text in enumerate(sub_items):
                keyword = item_text[:50].split("의")[0].split("에")[0].strip()
                rows.append(_make_rule(
                    rule_id   = f"usage_standard:art7:ho{ho_num}:item{sub_idx}",
                    rule_type = "allowed",
                    cat       = cat,
                    ho_num    = ho_num,
                    keyword   = keyword,
                    basis     = basis,
                    rule_text = item_text,
                    metadata  = {"ho_title": ho_title, "source_kind": "usage_standard"},
                ))

            rows.append(_make_rule(
                rule_id   = f"usage_standard:art7:ho{ho_num}:full",
                rule_type = "allowed",
                cat       = cat,
                ho_num    = ho_num,
                keyword   = ho_title[:50],
                basis     = basis,
                rule_text = ho_body.strip(),
                metadata  = {"ho_title": ho_title, "source_kind": "usage_standard", "is_full_ho": True},
            ))
            i += 2

    # 별표2 파싱
    bt2_m = re.search(
        r"(?:\[별표\s*2\]|별표\s*2(?=\s))(.+?)(?=\[별표\s*3\]|별표\s*3(?=\s)|$)",
        text, re.DOTALL
    )
    if bt2_m:
        bt2_body = bt2_m.group(1)
        d_items  = re.findall(r"\d+\.\s+(.{10,200}?)(?=\s+\d+\.\s+|\s*$)", bt2_body)
        if not d_items:
            d_items = re.findall(
                r"[가나다라마바사아자차카타파하]\.\s+(.{10,200}?)(?=\s+[가나다라마바사아자차카타파하]\.\s+|\s*$)",
                bt2_body
            )
        for d_idx, item_text in enumerate(d_items):
            item_text = item_text.strip()
            if len(item_text) < 5:
                continue
            rows.append(_make_rule(
                rule_id   = f"usage_standard:byultable2:item{d_idx}",
                rule_type = "disallowed",
                cat       = None,
                ho_num    = None,
                keyword   = item_text[:50],
                basis     = f"{_LAW_PREFIX} 별표 2",
                rule_text = item_text,
                metadata  = {"source_kind": "usage_standard"},
            ))

    return rows


def _make_rule(
    rule_id: str, rule_type: str, cat: str | None, ho_num: int | None,
    keyword: str, basis: str, rule_text: str, metadata: dict,
) -> dict[str, Any]:
    # 조/항/호 파싱
    art_m  = re.search(r"(제\d+조(?:의\d+)?)", basis)
    para_m = re.search(r"(제\d+항)", basis)
    item_m = re.search(r"(제\d+호)", basis)

    allowed: bool | None
    if rule_type == "allowed":
        allowed = True
    elif rule_type == "disallowed":
        allowed = False
    else:
        allowed = None

    return {
        "id":            rule_id,
        "source_name":   _SOURCE_NAME,
        "source_type":   "law",
        "source_path":   _USAGE_STANDARD_URL,
        "article_no":    art_m.group(1)  if art_m  else None,
        "paragraph_no":  para_m.group(1) if para_m else None,
        "item_no":       item_m.group(1) if item_m else None,
        "section_path":  basis,
        "chunk_id":      make_chunk_id(rule_id),
        "body":          rule_text,
        "record_type":   "rule",
        "content_type":  None,
        "rule_type":     rule_type,
        "category_code": cat,
        "category_name": _CAT_NAMES.get(cat, "") if cat else None,
        "allowed":       allowed,
        "limit_pct":     None,
        "keyword":       keyword,
        "item_pattern":  None,
        "legal_basis":   basis,
        "cited_laws":    _extract_cited_laws(rule_text),
        "keywords":      [],
        "hash":          hashlib.sha256(rule_text.encode()).hexdigest(),
        "metadata":      json.dumps(
            {**metadata, "category_number": ho_num},
            ensure_ascii=False,
        ),
    }


# ── profile rows 구성 (seed JSON → legal_rule_profiles) ──────────────────────

_DEFAULT_SEED_PATH = Path("scripts/seed_legal_rule_profiles.json")


def _build_profile_rows(seed_path: Path = _DEFAULT_SEED_PATH) -> list[dict[str, Any]]:
    """seed_legal_rule_profiles.json → legal_rule_profiles 테이블 레코드 목록."""
    if not seed_path.exists():
        log.warning("seed 파일 없음, profiles 적재 생략: %s", seed_path)
        return []

    config = json.loads(seed_path.read_text(encoding="utf-8"))
    rows: list[dict] = []

    # validator_synonyms → global scope, profile_key = 동의어 대표어
    for term, synonyms in config.get("validator_synonyms", {}).items():
        rows.append({
            "profile_id":    f"global:validator_synonym:{term}",
            "profile_scope": "global",
            "category_code": None,
            "profile_key":   term,
            "values_json":   json.dumps(synonyms, ensure_ascii=False),
            "metadata":      json.dumps({"original_scope": "validator_synonym"}, ensure_ascii=False),
        })

    # validator_profiles → category scope, profile_key = allow_terms / disallow_terms
    for cat_code, profile in config.get("validator_profiles", {}).items():
        for key, values in profile.items():
            rows.append({
                "profile_id":    f"category:{cat_code}:vp:{key}",
                "profile_scope": "category",
                "category_code": cat_code,
                "profile_key":   key,
                "values_json":   json.dumps(values, ensure_ascii=False),
                "metadata":      json.dumps({"original_scope": "validator_profile"}, ensure_ascii=False),
            })

    # classifier_profiles → category scope, profile_key = strong_terms / medium_terms / ...
    for cat_code, profile in config.get("classifier_profiles", {}).items():
        for key, values in profile.items():
            rows.append({
                "profile_id":    f"category:{cat_code}:cp:{key}",
                "profile_scope": "category",
                "category_code": cat_code,
                "profile_key":   key,
                "values_json":   json.dumps(values, ensure_ascii=False),
                "metadata":      json.dumps({"original_scope": "classifier_profile"}, ensure_ascii=False),
            })

    # generic_item_policies → item scope
    for key, policy in config.get("generic_item_policies", {}).items():
        rows.append({
            "profile_id":    f"item:policy:{key}",
            "profile_scope": "item",
            "category_code": None,
            "profile_key":   key,
            "values_json":   json.dumps(policy, ensure_ascii=False),
            "metadata":      json.dumps({"original_scope": "generic_item_policy"}, ensure_ascii=False),
        })

    return rows


def _extract_cited_laws(text: str) -> list[str]:
    found: set[str] = set()
    for pat in [r"제\d+조(?:의\d+)?(?:제\d+항(?:제\d+호(?:[가-하]목)?)?)?", r"별표\s*\d+(?:의\d+)?"]:
        found.update(re.findall(pat, text))
    return sorted(found)


# ── Qdrant 기존 포인트 삭제 ──────────────────────────────────────────────────

def _delete_usage_standard_points(collection_name: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="metadata.source_type", match=MatchValue(value="usage_standard"))
            ]),
        )
        log.info("기존 usage_standard 포인트 삭제 완료")
    except Exception as e:
        log.warning("usage_standard 삭제 실패 (무시): %s", e)


# ── 전체 파이프라인 ──────────────────────────────────────────────────────────

def run_usage_standard_pipeline(
    collection_name: str = DEFAULT_COLLECTION,
    database_url: str | None = None,
    skip_qdrant: bool = False,
    skip_rdb: bool = False,
    skip_profiles: bool = False,
    force_refresh: bool = False,
) -> dict[str, int]:
    """
    건설업 산안비 계상 및 사용기준 수집 → 조/항/호/목 파싱 → Qdrant + RDB 적재.

    Qdrant : legal_master rows(조·별표·rule) → Document, chunk_id=uuid5(master_id) 고정 upsert
    RDB    : legal_master(corpus+rule 통합) + legal_rule_profiles(seed JSON)

    skip_profiles=True : legal_rule_profiles 적재 건너뜀 (run_pipeline에서 PDF 단계가 이미 적재한 경우)
    force_refresh=True : 24시간 캐시를 무시하고 law.go.kr 재수집

    Returns: {"qdrant": n, "master": n, "corpus": n, "rules": n, "profiles": n}
    """
    from langchain_core.documents import Document as LCDocument

    log.info("산안비 사용기준 파이프라인 시작")

    raw_text = fetch_usage_standard(force_refresh=force_refresh)
    log.info("HTML 수집 완료: %d자", len(raw_text))

    # rows를 한 번만 빌드해서 Qdrant + RDB 양쪽에 재사용
    raw_master_rows = _build_corpus_rows(raw_text) + _build_rules_rows(raw_text)
    master_rows = _dedupe_rows_by_id(raw_master_rows)
    if len(master_rows) != len(raw_master_rows):
        log.info(
            "산안비 사용기준 row dedupe: %d개 → %d개",
            len(raw_master_rows),
            len(master_rows),
        )
    profile_rows = _build_profile_rows(_DEFAULT_SEED_PATH)

    qdrant_count = 0

    if not skip_qdrant:
        # legal_master rows → LangChain Document (chunk_id = Qdrant point ID)
        lc_docs: list[LCDocument] = []
        chunk_ids: list[str] = []
        for row in master_rows:
            breadcrumb = row.get("section_path") or row["source_name"]
            page_content = f"{breadcrumb}\n\n{row['body']}"
            lc_docs.append(LCDocument(
                page_content=page_content,
                metadata={
                    "source":      row["source_name"],
                    "source_type": "usage_standard",
                    "header_1":    _LAW_PREFIX,
                    "article_no":  row.get("article_no"),
                    "record_type": row["record_type"],
                    "master_id":   row["id"],
                    "chunk_id":    row["chunk_id"],
                },
            ))
            chunk_ids.append(row["chunk_id"])

        log.info("Document 변환 완료: %d개", len(lc_docs))
        _delete_usage_standard_points(collection_name)
        upsert_with_ids(collection_name=collection_name, documents=lc_docs, ids=chunk_ids)
        qdrant_count = len(lc_docs)
        log.info("Qdrant 적재 완료: %d개 (chunk_id 연결)", qdrant_count)

    master_count  = 0
    corpus_count  = 0
    rules_count   = 0
    profile_count = 0

    if not skip_rdb:
        rdb_result    = upsert_to_rdb(
            raw_text,
            database_url=database_url,
            _rows=master_rows,
            _profiles=[] if skip_profiles else profile_rows,
        )
        master_count  = rdb_result["master"]
        corpus_count  = rdb_result["corpus"]
        rules_count   = rdb_result["rules"]
        profile_count = rdb_result["profiles"]
        log.info(
            "적재 완료 → Qdrant: %d개, Master: %d개 (Corpus: %d, Rules: %d), Profiles: %d개",
            qdrant_count, master_count, corpus_count, rules_count, profile_count,
        )

    return {
        "qdrant":    qdrant_count,
        "master":    master_count,
        "corpus":    corpus_count,
        "rules":     rules_count,
        "profiles":  profile_count,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="산안비 사용기준 전문 → Qdrant + RDB 적재")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-rdb", action="store_true")
    parser.add_argument("--force-refresh", action="store_true",
                        help="24시간 캐시를 무시하고 law.go.kr 재수집")
    parser.add_argument("--dry-run", action="store_true",
                        help="마크다운 변환만 출력, 적재 없음 (캐시 우선 사용)")
    args = parser.parse_args()

    if args.dry_run:
        raw = fetch_usage_standard(force_refresh=args.force_refresh)
        md  = convert_to_markdown(raw)
        print(md)
        print(f"\n--- 총 {len(md)}자 ---")
    else:
        result = run_usage_standard_pipeline(
            collection_name=args.collection,
            skip_qdrant=args.skip_qdrant,
            skip_rdb=args.skip_rdb,
            force_refresh=args.force_refresh,
        )
        print(
            f"\n적재 완료 → "
            f"Qdrant: {result['qdrant']}개, "
            f"Master: {result['master']}개 "
            f"(Corpus: {result['corpus']} / Rules: {result['rules']}), "
            f"Profiles: {result['profiles']}개"
        )
