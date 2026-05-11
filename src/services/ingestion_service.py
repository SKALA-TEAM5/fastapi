# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 함수 정의 ]
#
# 1. run_pipeline() : 전체 인덱싱 파이프라인 (버전 체크 → Vector DB 적재)
# 2. process_pdf()  : PDF 법령 파일을 마크다운으로 변환 후 벡터 DB 적재
# 3. run_query()    : Agentic RAG 기반 법령 질의응답
# --------------------------------------------------------------------------
"""
RAG 인덱싱 파이프라인 + Agentic RAG 쿼리.

Public API:
    run_pipeline(...)   전체 인덱싱 파이프라인 (버전 체크 → Vector DB 적재)
    run_query(...)      Agentic RAG 질의응답

버전 체크 분기:
    web_date <= pdf_date  →  "web+pdf"  → 웹 + PDF 모두 Vector DB 저장
    web_date >  pdf_date  →  "web"      → 웹만 저장 (PDF보다 최신 법령)
    웹 스크래핑 실패      →  "pdf"      → PDF fallback

중복 실행 방지:
    .cache/indexing_state_{collection}.json 상태 + 웹 캐시 TTL 유효 시 스킵.
    force=True 로 강제 재인덱싱.

TODO(refresh):
    법령 개정 자동 반영이 필요해지면 이 모듈을 배치 파이프라인의 진입점으로 사용한다.
    권장 흐름은 최신 법령 감지 → 원문 수집 → 청킹/임베딩 → 저장소 갱신 → 이력 기록이다.
"""

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.judge import judge
from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import (
    DEFAULT_COLLECTION,
    LocalJSONCache,
    load_vectorstore,
    reset_collection,
    upsert_documents,
)
from src.schemas.shared import AgenticRAGState, AuditResult
from src.services.ingestion.breadcrumb import inject_breadcrumbs
from src.services.ingestion.converter import convert_pdf_to_markdown
from src.services.ingestion.restructure import restructure_markdown
from src.services.ingestion.splitter import split_markdown
from src.services.ingestion.web_scraper import LAW_URL, fetch_law_data

log = logging.getLogger(__name__)

_cache = LocalJSONCache()
_LAW_CACHE_KEY = "law_go_kr_산안비_고시"
_STATE_DIR = Path(".cache")


# ── 인덱싱 상태 관리 ─────────────────────────────────────────────

def _state_path(collection: str) -> Path:
    return _STATE_DIR / f"indexing_state_{collection}.json"


def _load_state(collection: str) -> dict | None:
    path = _state_path(collection)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_state(collection: str, result: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(collection).write_text(
        json.dumps(
            {**result, "indexed_at": datetime.now().isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── PDF 시행일 추출 ──────────────────────────────────────────────

def _pdf_effective_date(data_dir: str) -> date | None:
    dates: list[date] = []
    for pdf in Path(data_dir).glob("*.pdf"):
        m = re.search(r"(\d{4})(\d{2})(\d{2})", pdf.name)
        if m:
            try:
                dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
    return max(dates) if dates else None


# ── 웹 데이터 → Vector DB ────────────────────────────────────────

_ARTICLE_BOUNDARY = re.compile(r"(?=제\d+조(?:의\d+)?[\s(])")
_ARTICLE_NUM_RE = re.compile(r"제\d+조(?:의\d+)?(?:제\d+항(?:제\d+호(?:[가-하]목)?)?)?")


def _split_law_text(text: str) -> list[str]:
    """법령 조문(제X조) 단위로 1차 분리 후 800자 초과 시 추가 분할."""
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " "],
    )
    articles = [a.strip() for a in _ARTICLE_BOUNDARY.split(text) if a.strip()]
    result = []
    for article in articles:
        if len(article) <= 800:
            result.append(article)
        else:
            result.extend(char_splitter.split_text(article))
    return result


def _inject_web_cites(text: str) -> str:
    """청크 본문에 등장하는 모든 법령 조항 번호를 LEGAL_CITE 태그로 주입."""
    found = set(_ARTICLE_NUM_RE.findall(text))
    if not found:
        return text
    cite_str = " | ".join(sorted(found))
    return f"[LEGAL_CITE: {cite_str}]\n" + text


def _web_to_vector_db(web_data: dict, alert: str | None, collection: str) -> int:
    from langchain_core.documents import Document

    # TODO(refresh): PostgreSQL 연동 이후에는 청크 메타데이터 저장과 갱신 이력 기록도
    # 여기 또는 별도 repository 계층에서 함께 처리하도록 분리한다.
    meta = {
        "source": "web_법령",
        "source_url": LAW_URL,
        "effective_date": web_data["effective_date"],
        "alert": alert or "",
    }
    chunks = _split_law_text(web_data["content"])
    docs = [
        Document(page_content=_inject_web_cites(c), metadata=meta)
        for c in chunks if c.strip()
    ]
    upsert_documents(collection, docs)
    log.info(f"웹 청크 {len(docs)}개 → Vector DB 저장")
    return len(docs)


# ── 버전 체크 + 소스 결정 ────────────────────────────────────────

def _check_source(data_dir: str, collection: str, force: bool) -> dict:
    # TODO(refresh): 추후에는 로컬 캐시뿐 아니라 PostgreSQL에 저장된 법령 버전/시행일과도
    # 비교해 실제 재인덱싱 필요 여부를 판단한다.
    if not force:
        state = _load_state(collection)
        if state and _cache.get(_LAW_CACHE_KEY) is not None:
            log.info(f"이미 인덱싱됨 (소스: {state['source']}) — 스킵")
            return {**state, "skipped": True}

    web_data = _cache.get(_LAW_CACHE_KEY)
    if web_data is None:
        log.info("캐시 없음/만료 — 웹 스크래핑 시작")
        web_data = fetch_law_data()
        if web_data:
            _cache.set(_LAW_CACHE_KEY, web_data)

    pdf_date = _pdf_effective_date(data_dir)
    log.info(f"PDF 시행일: {pdf_date}")

    if web_data is None:
        log.warning("웹 스크래핑 실패 — PDF fallback")
        result = {
            "source": "pdf",
            "web_date": None,
            "pdf_date": str(pdf_date) if pdf_date else None,
            "alert": None,
            "chunks_added": 0,
            "skipped": False,
        }
        _save_state(collection, result)
        return result

    web_date = date.fromisoformat(web_data["effective_date"])
    log.info(f"웹 시행일: {web_date}")

    if pdf_date and web_date <= pdf_date:
        log.info(f"웹({web_date}) <= PDF({pdf_date}) — 웹+PDF 적재")
        chunks_added = _web_to_vector_db(web_data, alert=None, collection=collection)
        result = {
            "source": "web+pdf",
            "web_date": str(web_date),
            "pdf_date": str(pdf_date),
            "alert": None,
            "chunks_added": chunks_added,
            "skipped": False,
        }
    else:
        alert = (
            f"알림: {web_data['effective_date']} 기준 최신 법령을 적용함. "
            "기존 PDF 문서 업데이트 필요"
        )
        log.warning(alert)
        chunks_added = _web_to_vector_db(web_data, alert=alert, collection=collection)
        result = {
            "source": "web",
            "web_date": str(web_date),
            "pdf_date": str(pdf_date) if pdf_date else None,
            "alert": alert,
            "chunks_added": chunks_added,
            "skipped": False,
        }

    _save_state(collection, result)
    return result


# ── 단일 PDF 파이프라인 ───────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    output_dir: str = "outputs",
    collection: str = DEFAULT_COLLECTION,
    skip_vector_db: bool = False,
    reconvert: bool = False,
) -> list:
    """단일 PDF → markdown(final.md) → Vector DB.

    reconvert=False(기본): 기존 final.md가 있으면 변환 단계를 건너뛴다.
    """
    pdf = Path(pdf_path)
    out = Path(output_dir) / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "final.md"

    if not reconvert and final_path.exists():
        print(f"  → 기존 마크다운 재사용: {final_path}")
        final_md = final_path.read_text(encoding="utf-8")
    else:
        raw_md = convert_pdf_to_markdown(str(pdf))

        print("  계층 구조 재구성 중...")
        restructured_md = restructure_markdown(raw_md)

        print("  Breadcrumb 주입 중...")
        final_md = inject_breadcrumbs(restructured_md)

        final_path.write_text(final_md, encoding="utf-8")
        print(f"  → 마크다운 저장: {final_path}")

    print("  청킹 중...")
    chunks = split_markdown(final_md, source_metadata={"source": pdf.name, "source_stem": pdf.stem})
    print(f"  청크 {len(chunks)}개 생성")

    if not skip_vector_db:
        upsert_documents(collection, chunks)

    return chunks


# ── 전체 파이프라인 ──────────────────────────────────────────────

def run_pipeline(
    data_dir: str = "data",
    output_dir: str = "outputs",
    collection: str = DEFAULT_COLLECTION,
    force: bool = False,
    skip_vector_db: bool = False,
    reconvert: bool = False,
) -> dict:
    """
    완전한 인덱싱 파이프라인.

    반환:
        source, web_date, pdf_date, alert, chunks_added(웹),
        pdf_chunks, total_chunks, skipped

    force=True: Vector DB 컬렉션 초기화 후 전체 재인덱싱.
    reconvert=True: 기존 final.md를 무시하고 PDF 변환부터 재실행.

    TODO(refresh):
        scheduler/cron/job에서 이 함수를 호출하는 자동 갱신 엔트리포인트를 나중에 추가한다.
        지금은 수동 실행 기준으로 유지한다.
    """
    if force and not skip_vector_db:
        log.info("force=True — Vector DB 컬렉션 초기화")
        reset_collection(collection)
        _cache.invalidate(_LAW_CACHE_KEY)

    source_result = _check_source(
        data_dir=data_dir,
        collection=collection,
        force=force,
    )

    if source_result.get("skipped"):
        return {**source_result, "pdf_chunks": 0, "total_chunks": 0}

    pdf_chunks: list = []
    if source_result["source"] in ("pdf", "web+pdf"):
        pdf_files = sorted(Path(data_dir).glob("*.pdf"))
        for idx, pdf in enumerate(pdf_files, 1):
            print(f"\n  [{idx}/{len(pdf_files)}] {pdf.name}")
            chunks = process_pdf(
                pdf_path=str(pdf),
                output_dir=output_dir,
                collection=collection,
                skip_vector_db=skip_vector_db,
                reconvert=reconvert,
            )
            pdf_chunks.extend(chunks)

    return {
        **source_result,
        "pdf_chunks": len(pdf_chunks),
        "total_chunks": source_result["chunks_added"] + len(pdf_chunks),
    }


# ── Agentic RAG 쿼리 ─────────────────────────────────────────────

def run_query(question: str, collection: str = DEFAULT_COLLECTION) -> AuditResult:
    """
    산안비 법령 적합성 판정 수행 (단일 질문).

    흐름: Ensemble(BM25+Kiwi+Vector) → ReRank(bge-reranker-v2-m3) → Judge
    사용 전 llm_config.configure(llm) 호출 필요.
    """
    vectorstore = load_vectorstore(collection_name=collection)
    retriever = build_retriever(vectorstore, collection_name=collection, k=10)

    state: AgenticRAGState = {
        "question": question,
        "retrieved_docs": [],
        "judgment": None,
        "retry_count": 0,
    }

    state = retrieve(state, retriever)
    state = rerank(state)

    while not state["retrieved_docs"] and state.get("retry_count", 0) < MAX_RETRY:
        state = rewrite_query(state)
        state = retrieve(state, retriever)
        state = rerank(state)

    state = judge(state)
    return AuditResult(**state["judgment"])
