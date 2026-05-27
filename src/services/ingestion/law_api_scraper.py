# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-21
#
# [ 주요 함수 정의 ]
#
# 1. fetch_law_articles()   : 법제처 Open API로 특정 조문 수집
# 2. articles_to_documents(): 조문 리스트 → LangChain Document 청크 변환
# 3. run_law_api_pipeline() : 전체 수집 → 청킹 → Qdrant 적재 파이프라인
# --------------------------------------------------------------------------
"""
법제처 Open API (open.law.go.kr) 기반 조문 수집 모듈.

필요한 법령 조문만 지정해서 가져온 뒤 기존 Vector DB 컬렉션에 upsert한다.

사용법:
    API_KEY 환경변수 설정 후:
    python -m src.services.ingestion.law_api_pipeline

    또는 코드에서:
    from src.services.ingestion.law_api_scraper import run_law_api_pipeline
    run_law_api_pipeline(collection_name="safety_law")
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

import requests
from langchain_core.documents import Document

from src.core.storage import DEFAULT_COLLECTION, make_chunk_id, upsert_with_ids

log = logging.getLogger(__name__)

# ── 법제처 Open API 설정 ────────────────────────────────────────────────────
_BASE_URL = "https://www.law.go.kr/DRF"
_REQUEST_DELAY = 0.5  # 과부하 방지 (초)


def _api_key() -> str:
    """LAW_API_KEY를 호출 시점에 읽는다 (dotenv 로드 후에도 반영됨)."""
    return os.environ.get("LAW_API_KEY", "")

# ── 수집 대상 법령 및 조문 ──────────────────────────────────────────────────
# key: 법령명 (law.go.kr 검색어와 동일하게)
# value: 수집할 조문번호 리스트 (문자열, "128의2" 형태도 허용)
TARGET_LAWS: dict[str, list[str]] = {
    "산업안전보건법": [
        "17", "18",          # 안전관리자·보건관리자 선임
        "24",                # 산업안전보건위원회
        "29", "30", "31", "32",  # 안전보건교육
        "36",                # 위험성평가
        "42",                # 유해위험방지계획서
        "47",                # 안전보건진단
        "63", "64", "65", "66",  # 도급인 안전보건 조치 (하청 현장 산안비 적용 범위)
        "69", "72",          # 산안비 계상
        "73", "74",          # 건설재해예방전문지도기관
        "75",                # 노사협의체
        "76",                # 기계·기구 대여자 조치 (장비 임대 비용)
        "80", "81", "82", "83", "84", "85", "86",  # 유해위험기계 안전인증·자율안전확인
        "120", "121", "122", "123", "124", "125", "126", "127", "128", "129",  # 석면·작업환경·휴게시설·건강진단
    ],
    "산업안전보건법 시행령": [
        "15", "16",          # 안전관리자 선임 기준
        "18", "22",          # 안전관리자·보건관리자 업무
        "52", "53", "54", "55",  # 위험성평가 세부 기준
        "59", "60",          # 안전보건교육 기준
        "67", "68",          # 건설재해예방전문지도기관 기준 (CAT_07)
        "74", "77",          # 보호구
    ],
    "산업안전보건법 시행규칙": [
        "29", "30", "31", "32", "33", "34", "35",  # 안전보건교육 세부 기준 (CAT_05)
        "89",                # 사용방법
        "97", "98", "99", "100",  # 보호구 지급 기준 (CAT_03)
    ],
    "중대재해 처벌 등에 관한 법률 시행령": [
        "4",                 # 안전보건관리체계 구축 의무
    ],
    "건설기술 진흥법": [
        "2", "62의3",        # 감리자 정의 + 스마트 안전장비
    ],
    "건축법": [
        "2",                 # 감리자 범위
    ],
    "응급의료에 관한 법률": [
        "14",                # 응급처치 교육비
    ],
    "감염병의 예방 및 관리에 관한 법률": [
        "2",                 # 마스크·손소독제 비용
    ],
}

# key: 법령명, value: 수집할 별표 번호
TARGET_LAW_APPENDICES: dict[str, list[str]] = {
    "산업안전보건법 시행규칙": [
        "4", "5",            # 안전보건교육 시간·내용
    ],
}

# key: 행정규칙명 검색어, value: 수집할 조문번호 리스트.
# None이면 해당 행정규칙의 조문형식 본문 전체를 수집한다.
TARGET_ADMIN_RULES: dict[str, list[str] | None] = {
    "산업재해예방시설자금 융자금 지원사업 및 보조금 지급사업 운영규정": [
        "2",                 # 스마트안전장비 지원사업 정의
    ],
    "예정가격 작성기준": [
        "15", "17", "18", "19",  # 공사원가·재료비·노무비·경비
    ],
    "지방자치단체 입찰 및 계약집행기준": None,  # API 본문 제공 시 전체 수집
}


# ── API 호출 헬퍼 ────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict:
    """법제처 API GET 요청. 실패 시 빈 딕셔너리 반환."""
    params.update({"OC": _api_key(), "type": "JSON"})
    try:
        resp = requests.get(f"{_BASE_URL}/{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("법제처 API 요청 실패: endpoint=%s params=%s err=%s", endpoint, params, e)
        return {}


def _stable_slug(value: str) -> str:
    slug = re.sub(r"[^\w가-힣]+", "_", value.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


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


def _clean_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return "\n".join(line.strip() for line in value.splitlines()).strip()


def _canonical_article_no(no: str) -> str:
    no = re.sub(r"\s+", "", str(no or ""))
    match = re.search(r"제?(\d+(?:의\d+)?)조?", no)
    return match.group(1) if match else no


def _appendix_no(value: str) -> str:
    value = str(value or "").strip()
    match = re.search(r"별표\s*(\d+(?:의\d+)?)", value)
    if match:
        parts = match.group(1).split("의", 1)
        base = parts[0].lstrip("0") or "0"
        return base + (f"의{parts[1].lstrip('0') or '0'}" if len(parts) > 1 else "")
    digits = re.sub(r"\D", "", value).lstrip("0")
    return digits or value


def _find_mst(law_name: str) -> str | None:
    """법령명으로 MST(법령 고유번호) 조회."""
    data = _get("lawSearch.do", {"target": "law", "query": law_name, "display": 10})
    laws = (data.get("LawSearch") or {}).get("law") or []
    if isinstance(laws, dict):
        laws = [laws]
    # 정확히 일치하는 것 우선
    for law in laws:
        if law.get("법령명한글") == law_name:
            return str(law["법령일련번호"])
    if laws:
        return str(laws[0]["법령일련번호"])
    return None


def _find_admrul_seq(query: str) -> str | None:
    """행정규칙명으로 현행 행정규칙 일련번호 조회."""
    data = _get(
        "lawSearch.do",
        {
            "target": "admrul",
            "query": query,
            "display": 10,
            "mobileYn": "Y",
            "nw": "1",
            "sort": "efdes",
        },
    )
    rules = (data.get("AdmRulSearch") or {}).get("admrul") or []
    if isinstance(rules, dict):
        rules = [rules]
    for rule in rules:
        if rule.get("행정규칙명") == query:
            return str(rule["행정규칙일련번호"])
    if rules:
        return str(rules[0]["행정규칙일련번호"])
    return None


def _fetch_law_by_mst(mst: str) -> tuple[str, list[dict], list[dict], dict[str, Any]]:
    """MST로 법령 전문 조회 → (법령명, 조문 리스트, 별표 리스트, 기본정보) 반환."""
    data = _get("lawService.do", {"target": "law", "MST": mst})
    law_data = data.get("법령") or {}
    basic_info = law_data.get("기본정보") or {}
    law_name = basic_info.get("법령명한글") or basic_info.get("법령명_한글") or ""
    articles_raw = (law_data.get("조문") or {}).get("조문단위") or []
    if isinstance(articles_raw, dict):
        articles_raw = [articles_raw]
    appendices_raw = (law_data.get("별표") or {}).get("별표단위") or []
    if isinstance(appendices_raw, dict):
        appendices_raw = [appendices_raw]
    return law_name, articles_raw, appendices_raw, basic_info


def _fetch_admrul_by_seq(admrul_seq: str) -> tuple[str, list[str], list[dict], dict[str, Any]]:
    """행정규칙 일련번호로 본문 조회 → (행정규칙명, 조문내용, 별표 리스트, 기본정보) 반환."""
    data = _get("lawService.do", {"target": "admrul", "ID": admrul_seq})
    service = data.get("AdmRulService") or {}
    basic_info = service.get("행정규칙기본정보") or {}
    rule_name = basic_info.get("행정규칙명", "")
    article_lines = service.get("조문내용") or []
    if isinstance(article_lines, str):
        article_lines = [article_lines] if article_lines.strip() else []
    appendices_raw = (service.get("별표") or {}).get("별표단위") or []
    if isinstance(appendices_raw, dict):
        appendices_raw = [appendices_raw]
    return rule_name, article_lines, appendices_raw, basic_info


def _normalize_article_no(no: str) -> str:
    """조문번호 정규화: 공백 제거, 소문자."""
    return _canonical_article_no(no).lower()


def _make_law_article_record(
    *,
    law_name: str,
    article_no: str,
    article_title: str,
    content: str,
    mst: str,
    basic_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_kind": "law",
        "law_name": law_name,
        "article_no": _canonical_article_no(article_no),
        "article_title": article_title.strip(),
        "content": _clean_text(content),
        "content_type": "article",
        "version_id": mst,
        "mst": mst,
        "admrul_seq": None,
        "stable_source_key": _stable_slug(law_name),
        "source_path": f"https://www.law.go.kr/DRF/lawService.do?target=law&MST={mst}&type=JSON",
        "metadata": {
            "source_kind": "law",
            "mst": mst,
            "law_id": basic_info.get("법령ID"),
            "effective_date": basic_info.get("시행일자"),
            "promulgation_no": basic_info.get("공포번호"),
            "article_title": article_title.strip(),
        },
    }


def _make_appendix_record(
    *,
    source_kind: str,
    law_name: str,
    appendix: dict[str, Any],
    version_id: str,
    source_path: str,
    basic_info: dict[str, Any],
) -> dict[str, Any] | None:
    appendix_no = _appendix_no(
        str(appendix.get("별표구분") or "별표") + " " + str(appendix.get("별표번호") or "")
    )
    branch_no = str(appendix.get("별표가지번호") or "").strip()
    if branch_no and branch_no != "00":
        appendix_no = f"{appendix_no}의{int(branch_no)}"
    appendix_label = f"{appendix.get('별표구분') or '별표'} {appendix_no}".strip()
    title = str(appendix.get("별표제목") or "").strip()
    content = _clean_text("\n".join(_flatten_text(appendix.get("별표내용"))))
    if not content:
        return None

    return {
        "source_kind": source_kind,
        "law_name": law_name,
        "article_no": appendix_label,
        "article_title": title,
        "content": content,
        "content_type": "guideline",
        "version_id": version_id,
        "mst": version_id if source_kind == "law" else None,
        "admrul_seq": version_id if source_kind == "admrul" else None,
        "stable_source_key": _stable_slug(law_name),
        "source_path": source_path,
        "metadata": {
            "source_kind": source_kind,
            "mst": version_id if source_kind == "law" else None,
            "admrul_seq": version_id if source_kind == "admrul" else None,
            "law_id": basic_info.get("법령ID") or basic_info.get("행정규칙ID"),
            "effective_date": basic_info.get("시행일자"),
            "promulgation_no": basic_info.get("공포번호") or basic_info.get("발령번호"),
            "article_title": title,
            "appendix_key": appendix.get("별표키"),
            "appendix_type": appendix.get("별표구분"),
        },
    }


def _make_admrul_article_record(
    *,
    rule_name: str,
    raw_article: str,
    admrul_seq: str,
    basic_info: dict[str, Any],
) -> dict[str, Any] | None:
    match = re.match(r"제(\d+(?:의\d+)?)조\s*\(([^)]+)\)\s*(.*)", raw_article.strip(), re.DOTALL)
    if not match:
        return None
    article_no = match.group(1)
    title = match.group(2).strip()
    body = match.group(3).strip()
    if not body:
        return None
    return {
        "source_kind": "admrul",
        "law_name": rule_name,
        "article_no": article_no,
        "article_title": title,
        "content": _clean_text(body),
        "content_type": "article",
        "version_id": admrul_seq,
        "mst": None,
        "admrul_seq": admrul_seq,
        "stable_source_key": _stable_slug(rule_name),
        "source_path": (
            "https://www.law.go.kr/DRF/lawService.do"
            f"?target=admrul&ID={admrul_seq}&type=JSON"
        ),
        "metadata": {
            "source_kind": "admrul",
            "admrul_seq": admrul_seq,
            "law_id": basic_info.get("행정규칙ID"),
            "effective_date": basic_info.get("시행일자"),
            "promulgation_no": basic_info.get("발령번호"),
            "article_title": title,
        },
    }


def fetch_law_articles(
    target_laws: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    법제처 Open API로 지정한 조문을 수집한다.

    Returns:
        list of dicts with keys:
            law_name, article_no, article_title, content, mst, law_type
    """
    if not _api_key():
        raise RuntimeError(
            "LAW_API_KEY 환경변수가 설정되지 않았습니다. "
            "open.law.go.kr에서 발급 후 설정하세요."
        )

    targets = target_laws or TARGET_LAWS
    result: list[dict[str, Any]] = []

    for law_name, article_nos in targets.items():
        log.info("수집 시작: %s (조문 %d개)", law_name, len(article_nos))

        mst = _find_mst(law_name)
        if not mst:
            log.warning("MST 조회 실패: %s", law_name)
            continue

        time.sleep(_REQUEST_DELAY)
        fetched_law_name, all_articles, all_appendices, basic_info = _fetch_law_by_mst(mst)
        display_name = fetched_law_name or law_name

        target_nos = {_normalize_article_no(no) for no in article_nos}

        for article in all_articles:
            raw_no = str(article.get("조문번호", ""))
            branch_no = str(article.get("조문가지번호") or "").strip()
            if branch_no and branch_no != "00":
                raw_no = f"{raw_no}의{int(branch_no)}"
            if _normalize_article_no(raw_no) not in target_nos:
                continue

            # 항내용을 합쳐서 전문 구성
            content_parts = []
            paragraphs = article.get("항") or []
            if isinstance(paragraphs, dict):
                paragraphs = [paragraphs]
            for para in paragraphs:
                if isinstance(para, dict):
                    content_parts.append(str(para.get("항내용") or "").strip())

            # 항이 없으면 조문내용 사용
            if not content_parts:
                raw = str(article.get("조문내용") or "").strip()
                if raw:
                    content_parts.append(raw)

            content = "\n".join(p for p in content_parts if p).strip()
            if not content:
                content = str(article.get("조문내용") or "").strip()
            if not content:
                continue

            result.append(
                _make_law_article_record(
                    law_name=display_name,
                    article_no=raw_no,
                    article_title=str(article.get("조문제목") or "").strip(),
                    content=content,
                    mst=mst,
                    basic_info=basic_info,
                )
            )

        appendix_targets = {_appendix_no(no) for no in TARGET_LAW_APPENDICES.get(law_name, [])}
        if appendix_targets:
            for appendix in all_appendices:
                if str(appendix.get("별표구분") or "별표").strip() != "별표":
                    continue
                appendix_no = _appendix_no(str(appendix.get("별표번호") or ""))
                branch_no = str(appendix.get("별표가지번호") or "").strip()
                if branch_no and branch_no != "00":
                    appendix_no = f"{appendix_no}의{int(branch_no)}"
                if appendix_no not in appendix_targets:
                    continue
                record = _make_appendix_record(
                    source_kind="law",
                    law_name=display_name,
                    appendix=appendix,
                    version_id=mst,
                    source_path=f"https://www.law.go.kr/DRF/lawService.do?target=law&MST={mst}&type=JSON",
                    basic_info=basic_info,
                )
                if record:
                    result.append(record)

        log.info("  → %s: %d개 조문 수집 완료", display_name, len([r for r in result if r["mst"] == mst]))
        time.sleep(_REQUEST_DELAY)

    for query, article_nos in TARGET_ADMIN_RULES.items():
        log.info("행정규칙 수집 시작: %s", query)
        admrul_seq = _find_admrul_seq(query)
        if not admrul_seq:
            log.warning("행정규칙 일련번호 조회 실패: %s", query)
            continue

        time.sleep(_REQUEST_DELAY)
        fetched_rule_name, raw_articles, all_appendices, basic_info = _fetch_admrul_by_seq(admrul_seq)
        display_name = fetched_rule_name or query
        target_nos = None if article_nos is None else {_normalize_article_no(no) for no in article_nos}

        added = 0
        for raw_article in raw_articles:
            record = _make_admrul_article_record(
                rule_name=display_name,
                raw_article=raw_article,
                admrul_seq=admrul_seq,
                basic_info=basic_info,
            )
            if not record:
                continue
            if target_nos is not None and _normalize_article_no(record["article_no"]) not in target_nos:
                continue
            result.append(record)
            added += 1

        # 조문형식이 아닌 행정규칙이 API 본문을 제공하지 않는 경우가 있어 별표가 있으면 함께 수집한다.
        for appendix in all_appendices:
            record = _make_appendix_record(
                source_kind="admrul",
                law_name=display_name,
                appendix=appendix,
                version_id=admrul_seq,
                source_path=(
                    "https://www.law.go.kr/DRF/lawService.do"
                    f"?target=admrul&ID={admrul_seq}&type=JSON"
                ),
                basic_info=basic_info,
            )
            if record and target_nos is None:
                result.append(record)
                added += 1

        log.info("  → %s: %d개 행정규칙 조문 수집 완료", display_name, added)
        time.sleep(_REQUEST_DELAY)

    log.info("전체 %d개 조문 수집 완료", len(result))
    return result


def _law_article_row_id(article: dict[str, Any]) -> str:
    content_type = article.get("content_type") or "article"
    kind = "appendix" if content_type == "guideline" else "article"
    source_key = article.get("stable_source_key") or _stable_slug(str(article.get("law_name", "")))
    section_key = re.sub(r"\s+", "", str(article.get("article_no", "")))
    return f"law_api:{source_key}:{kind}:{section_key}"


def _dedupe_law_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qdrant/RDB 모두 같은 master_id 기준으로 적재되도록 조문을 dedup한다."""
    seen: dict[str, dict[str, Any]] = {}
    for article in articles:
        row_id = _law_article_row_id(article)
        if row_id in seen:
            log.warning("중복 조문 dedup: id=%s (동일 MST가 여러 법령명으로 수집됨)", row_id)
        seen[row_id] = article
    return list(seen.values())


def _section_heading(article: dict[str, Any]) -> str:
    article_no = str(article["article_no"])
    article_title = str(article.get("article_title") or "").strip()
    if article.get("content_type") == "guideline":
        return article_no + (f" ({article_title})" if article_title else "")
    return f"제{article_no}조" + (f" ({article_title})" if article_title else "")


def _legal_basis(article: dict[str, Any]) -> str:
    return f"{article['law_name']} {_section_heading(article)}"


# ── Document 변환 ────────────────────────────────────────────────────────────

def articles_to_documents(articles: list[dict[str, Any]]) -> list[Document]:
    """
    수집된 조문 리스트를 LangChain Document로 변환한다.

    메타데이터 구조는 기존 ingestion 파이프라인과 동일하게 맞춘다:
      source, header_1, header_2, breadcrumb, source_type
    """
    docs: list[Document] = []

    for article in articles:
        law_name = article["law_name"]
        article_no = article["article_no"]
        content = article["content"]

        row_id   = _law_article_row_id(article)
        chunk_id = make_chunk_id(row_id)

        # 헤더 구성 (기존 청크와 동일한 메타 키 사용)
        h2         = _section_heading(article)
        breadcrumb = f"{law_name} > {h2}"

        # LEGAL_CITE 태그 삽입 (citation vote에서 활용)
        cite_tag     = f"[LEGAL_CITE: {_legal_basis(article)}]"
        page_content = f"{breadcrumb}\n\n{content}\n\n{cite_tag}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source":       f"법제처 Open API — {law_name} 제{article_no}조",
                    "header_1":     law_name,
                    "header_2":     h2,
                    "breadcrumb":   breadcrumb,
                    "source_type":  "law_article",
                    "record_type":  "corpus",
                    "law_name":     law_name,
                    "article_no":   article_no,
                    "mst":          article.get("mst"),
                    "admrul_seq":   article.get("admrul_seq"),
                    "content_type": article.get("content_type", "article"),
                    "master_id":    row_id,
                    "chunk_id":     chunk_id,   # ← Qdrant point ID와 동일
                },
            )
        )

    return docs


# ── 전체 파이프라인 ──────────────────────────────────────────────────────────

def _delete_law_articles(collection_name: str) -> None:
    """기존 law_article 포인트를 모두 삭제 (중복 방지)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    import os
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_type",
                        match=MatchValue(value="law_article"),
                    )
                ]
            ),
        )
        log.info("기존 law_article 포인트 삭제 완료 (collection=%s)", collection_name)
    except Exception as e:
        log.warning("law_article 삭제 실패 (무시): %s", e)


def law_article_to_row(article: dict[str, Any]) -> dict[str, Any]:
    """수집된 법령/행정규칙 조문을 legal_master row dict로 변환한다."""
    row_id = _law_article_row_id(article)
    chunk_id = make_chunk_id(row_id)
    h2 = _section_heading(article)
    legal_basis = _legal_basis(article)
    content = article.get("content", "")
    meta = {
        "source_type": "law_article",
        "source_kind": article.get("source_kind"),
        "article_title": article.get("article_title", ""),
        "mst": article.get("mst"),
        "admrul_seq": article.get("admrul_seq"),
        "version_id": article.get("version_id"),
        "chunk_id": chunk_id,
        **(article.get("metadata") or {}),
    }

    return {
        "id": row_id,
        "source_name": article["law_name"],
        "source_type": "law",
        "source_path": article.get("source_path"),
        "article_no": h2,
        "paragraph_no": None,
        "item_no": None,
        "section_path": f"{article['law_name']} > {h2}",
        "chunk_id": chunk_id,
        "body": content,
        "record_type": "corpus",
        "content_type": article.get("content_type", "article"),
        "rule_type": None,
        "category_code": None,
        "category_name": None,
        "allowed": None,
        "limit_pct": None,
        "keyword": None,
        "item_pattern": None,
        "legal_basis": legal_basis,
        "cited_laws": [legal_basis],
        "keywords": [article["law_name"], h2],
        "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "metadata": json.dumps(meta, ensure_ascii=False),
    }


def _upsert_articles_to_rdb(articles: list[dict[str, Any]], database_url: str) -> int:
    """수집된 조문 리스트를 legal_master (record_type='corpus') 에 적재.

    law_name 기준으로 source-scoped DELETE 후 INSERT.
    Returns: 적재된 레코드 수.
    """
    import psycopg2
    from psycopg2.extras import Json, execute_values

    records: list[tuple] = []
    for art in articles:
        row = law_article_to_row(art)
        if not row["body"]:
            continue

        records.append((
            row["id"],
            row["source_name"],
            row["source_type"],
            row["source_path"],
            row["article_no"],
            row["paragraph_no"],
            row["item_no"],
            row["section_path"],
            row["chunk_id"],
            row["body"],
            row["record_type"],
            row["content_type"],
            row["rule_type"],
            row["category_code"],
            row["category_name"],
            row["allowed"],
            row["limit_pct"],
            row["keyword"],
            row["item_pattern"],
            row["legal_basis"],
            row["cited_laws"],
            row["keywords"],
            row["hash"],
            Json(json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]),
        ))

    if not records:
        return 0

    source_names = list({r[1] for r in records})

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET search_path TO legal_rag, public")
                # source-scoped DELETE
                cur.execute(
                    "DELETE FROM legal_master WHERE source_name = ANY(%s) AND source_type = 'law'",
                    (source_names,),
                )
                execute_values(
                    cur,
                    """
                    INSERT INTO legal_master (
                        id, source_name, source_type, source_path,
                        article_no, paragraph_no, item_no, section_path,
                        chunk_id, body, record_type, content_type, rule_type,
                        category_code, category_name, allowed, limit_pct,
                        keyword, item_pattern, legal_basis,
                        cited_laws, keywords, hash, metadata
                    ) VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        body     = EXCLUDED.body,
                        hash     = EXCLUDED.hash,
                        metadata = EXCLUDED.metadata
                    """,
                    records,
                )
    finally:
        conn.close()

    log.info("RDB 적재 완료: %d개 조문 → legal_master", len(records))
    return len(records)


def run_law_api_pipeline(
    collection_name: str = DEFAULT_COLLECTION,
    target_laws: dict[str, list[str]] | None = None,
    database_url: str | None = None,
) -> dict[str, int]:
    """
    법제처 Open API 수집 → Document 변환 → Qdrant upsert 전체 파이프라인.

    기존 law_article 포인트를 삭제 후 재적재하므로 중복 없이 항상 최신 상태 유지.
    database_url 이 지정되면 legal_master(RDB) 에도 동시 적재한다.

    Returns:
        {"qdrant": n, "rdb": n}
    """
    log.info("법제처 Open API 파이프라인 시작 (collection=%s)", collection_name)

    articles = _dedupe_law_articles(fetch_law_articles(target_laws=target_laws))
    if not articles:
        log.warning("수집된 조문이 없습니다.")
        return {"qdrant": 0, "rdb": 0}

    docs = articles_to_documents(articles)
    log.info("Document 변환 완료: %d개", len(docs))

    # chunk_id 목록 추출 (documents의 metadata에서 가져옴)
    chunk_ids = [doc.metadata["chunk_id"] for doc in docs]

    # 기존 law_article 중복 제거 후 재적재 (chunk_id 고정 → refresh 시 포인트 특정 가능)
    _delete_law_articles(collection_name)

    upsert_with_ids(collection_name=collection_name, documents=docs, ids=chunk_ids)
    log.info("Qdrant upsert 완료: collection=%s, docs=%d (chunk_id 연결)", collection_name, len(docs))

    # RDB 적재 (database_url 이 지정된 경우)
    rdb_count = 0
    if database_url:
        try:
            rdb_count = _upsert_articles_to_rdb(articles, database_url)
            log.info("법제처 Open API RDB 적재 완료: %d개", rdb_count)
        except Exception as e:
            log.warning("법제처 Open API RDB 적재 실패 (Qdrant 적재는 완료됨): %s", e)

    return {"qdrant": len(docs), "rdb": rdb_count}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="법제처 Open API → Qdrant 적재")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    result = run_law_api_pipeline(collection_name=args.collection)
    print(f"\n적재 완료 → Qdrant: {result['qdrant']}개, RDB: {result['rdb']}개")
