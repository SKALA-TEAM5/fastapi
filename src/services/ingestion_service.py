# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-05-22
#
# [ 주요 함수 정의 ]
#
# 1. run_pipeline() : 전체 인덱싱 파이프라인
# 2. process_pdf()  : PDF 법령 파일을 마크다운으로 변환 후 벡터 DB 적재
# 3. run_query()    : Agentic RAG 기반 법령 질의응답
# --------------------------------------------------------------------------
"""
RAG 인덱싱 파이프라인 + Agentic RAG 쿼리.

Public API:
    run_pipeline(...)   전체 인덱싱 파이프라인
    run_query(...)      Agentic RAG 질의응답

파이프라인 구성:
    1. PDF 법령 파일 → 마크다운 변환 → Qdrant 청크
    2. 법제처 Open API 조문 수집 → Qdrant (law_article)
    3. 산안비 사용기준 고시 파싱 → Qdrant (usage_standard) + RDB (legal_rules)

중복 실행 방지:
    .cache/indexing_state_{collection}.json 상태 파일 유효 시 스킵.
    force=True 로 강제 재인덱싱.

TODO(refresh):
    법령 개정 자동 반영이 필요해지면 이 모듈을 배치 파이프라인의 진입점으로 사용한다.
    권장 흐름은 최신 법령 감지 → 원문 수집 → 청킹/임베딩 → 저장소 갱신 → 이력 기록이다.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 또는 상위 디렉토리의 .env 자동 탐색 (skala 모노레포 구조 대응)
load_dotenv()

from src.core.judge import judge
from src.core.rag import MAX_RETRY, build_retriever, rerank, retrieve, rewrite_query
from src.core.storage import (
    DEFAULT_COLLECTION,
    load_vectorstore,
    reset_collection,
)
from src.schemas.shared import AgenticRAGState, AuditResult
from src.services.ingestion.breadcrumb import inject_breadcrumbs
from src.services.ingestion.converter import convert_pdf_to_markdown
from src.services.ingestion.law_api_scraper import run_law_api_pipeline
from src.services.ingestion.restructure import restructure_markdown
from src.services.ingestion.usage_standard_scraper import run_usage_standard_pipeline

log = logging.getLogger(__name__)

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


# ── 단일 PDF 파이프라인 ───────────────────────────────────────────


def process_pdf(
    pdf_path: str,
    output_dir: str = "outputs",
    reconvert: bool = False,
) -> Path:
    """단일 PDF → markdown(final.md) 변환만 수행. Qdrant/RDB 적재는 run_pipeline에서 일괄 처리.

    reconvert=False(기본): 기존 final.md가 있으면 변환 단계를 건너뛴다.
    반환: final.md 경로
    """
    pdf = Path(pdf_path)
    out = Path(output_dir) / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "final.md"

    if not reconvert and final_path.exists():
        print(f"  → 기존 마크다운 재사용: {final_path}")
        return final_path

    raw_md = convert_pdf_to_markdown(str(pdf))

    print("  계층 구조 재구성 중...")
    restructured_md = restructure_markdown(raw_md)

    print("  Breadcrumb 주입 중...")
    final_md = inject_breadcrumbs(restructured_md)

    final_path.write_text(final_md, encoding="utf-8")
    print(f"  → 마크다운 저장: {final_path}")
    return final_path


# ── 전체 파이프라인 ──────────────────────────────────────────────


def run_pipeline(
    data_dir: str = "data",
    output_dir: str = "outputs",
    collection: str = DEFAULT_COLLECTION,
    force: bool = False,
    skip_vector_db: bool = False,
    reconvert: bool = False,
    skip_law_api: bool = False,
    skip_usage_standard: bool = False,
    database_url: str | None = None,
) -> dict:
    """
    완전한 인덱싱 파이프라인.

    반환:
        pdf_chunks, total_chunks, skipped,
        law_api_chunks, usage_standard_qdrant, usage_standard_rdb,
        pdf_rdb_master, pdf_rdb_corpus, pdf_rdb_rules, pdf_rdb_profiles,
        law_api_rdb

    force=True: Vector DB 컬렉션 초기화 후 전체 재인덱싱.
    reconvert=True: 기존 final.md를 무시하고 PDF 변환부터 재실행.
    skip_law_api=True: 법제처 Open API 조문 수집 건너뜀.
    skip_usage_standard=True: 산안비 사용기준 고시 수집/RDB 적재 건너뜀.
    database_url: PostgreSQL 연결 문자열. 지정 시 PDF·법제처 조문도 legal_master에 적재.
                  미지정 시 환경변수 DATABASE_URL 을 자동으로 사용한다.

    TODO(refresh):
        scheduler/cron/job에서 이 함수를 호출하는 자동 갱신 엔트리포인트를 나중에 추가한다.
        지금은 수동 실행 기준으로 유지한다.
    """
    db_url = database_url or os.environ.get(
        "DATABASE_URL",
        "postgresql://safety_user:safety_password@localhost:5432/safety",
    )
    if force and not skip_vector_db:
        log.info("force=True — Vector DB 컬렉션 초기화")
        reset_collection(collection)

    # 중복 실행 방지 (force=False 일 때)
    if not force:
        state = _load_state(collection)
        if state:
            log.info("이미 인덱싱됨 — 스킵 (force=True 로 재실행 가능)")
            return {**state, "skipped": True}

    # ── PDF 적재 ─────────────────────────────────────────────────
    pdf_chunks: list = []
    pdf_rdb_master = 0
    pdf_rdb_corpus = 0
    pdf_rdb_rules = 0
    pdf_rdb_profiles = 0
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    # ── PDF: markdown 변환 (Qdrant/RDB 적재는 build_payload로 일괄) ──
    if pdf_files:
        print(f"\n  [PDF] {len(pdf_files)}개 파일 마크다운 변환 시작")
        for idx, pdf in enumerate(pdf_files, 1):
            print(f"\n  [{idx}/{len(pdf_files)}] {pdf.name}")
            process_pdf(pdf_path=str(pdf), output_dir=output_dir, reconvert=reconvert)

    # ── PDF → Qdrant + RDB (build_payload 기반, chunk_id 1:1 연결) ──
    if pdf_files:
        try:
            from langchain_core.documents import Document as LCDocument

            from src.core.storage import make_chunk_id, upsert_with_ids
            from src.repositories.legal_rules_exporter import (
                build_payload,
                execute_payload_to_rdb,
            )

            print("\n  [PDF → Qdrant + RDB] build_payload 파싱 중...")
            payload = build_payload(Path(output_dir))

            # corpus + rule 행 → LangChain Document (chunk_id = uuid5(master_id))
            lc_docs: list[LCDocument] = []
            chunk_ids: list[str] = []
            for row in payload.get("master", []):
                if row.get("record_type") not in ("corpus", "rule"):
                    continue
                master_id = row["master_id"]
                chunk_id = make_chunk_id(master_id)
                breadcrumb = row.get("section_path") or row.get("source_name", "")
                body = row.get("body") or ""
                page_content = f"{breadcrumb}\n\n{body}" if breadcrumb else body
                lc_docs.append(
                    LCDocument(
                        page_content=page_content,
                        metadata={
                            "source": row.get("source_name", ""),
                            "source_type": row.get("source_type", ""),
                            "record_type": row.get("record_type", ""),
                            "article_no": row.get("article_no"),
                            "section_path": row.get("section_path"),
                            "master_id": master_id,
                            "chunk_id": chunk_id,
                        },
                    )
                )
                chunk_ids.append(chunk_id)
                # master_id에 chunk_id를 역으로 기록 (RDB execute_payload_to_rdb에서 활용 가능하도록)
                row["_chunk_id"] = chunk_id

            pdf_chunks = lc_docs  # type: ignore[assignment]

            if not skip_vector_db and lc_docs:
                upsert_with_ids(
                    collection_name=collection,
                    documents=lc_docs,
                    ids=chunk_ids,
                )
                print(f"  [PDF → Qdrant] 완료: {len(lc_docs)}개 (chunk_id 연결)")

            if db_url:
                rdb = execute_payload_to_rdb(payload, db_url)
                pdf_rdb_master = rdb["master"]
                pdf_rdb_corpus = rdb["corpus"]
                pdf_rdb_rules = rdb["rules"]
                pdf_rdb_profiles = rdb["profiles"]
                print(
                    f"  [PDF → RDB] 완료: "
                    f"Master {pdf_rdb_master}개 (Corpus {pdf_rdb_corpus} / Rules {pdf_rdb_rules}), "
                    f"Profiles {pdf_rdb_profiles}개"
                )
        except Exception as e:
            log.warning("PDF Qdrant+RDB 적재 실패 (건너뜀): %s", e, exc_info=True)

    # ── 법제처 Open API 조문 (Qdrant + RDB) ─────────────────────
    law_api_chunks = 0
    law_api_rdb = 0
    if not skip_law_api and not skip_vector_db:
        try:
            print("\n  [법제처 Open API] 조문 수집 및 Qdrant+RDB 적재 시작...")
            law_api_result = run_law_api_pipeline(
                collection_name=collection,
                database_url=db_url,
            )
            law_api_chunks = law_api_result["qdrant"]
            law_api_rdb = law_api_result["rdb"]
            print(
                f"  [법제처 Open API] 완료: "
                f"Qdrant {law_api_chunks}개, RDB {law_api_rdb}개"
            )
        except Exception as e:
            log.warning("법제처 Open API 파이프라인 실패 (건너뜀): %s", e)

    # ── 산안비 사용기준 고시 (Qdrant + RDB) ─────────────────────
    usage_std_qdrant = 0
    usage_std_master = 0
    usage_std_corpus = 0
    usage_std_rules = 0
    usage_std_profiles = 0
    if not skip_usage_standard:
        try:
            print("\n  [산안비 사용기준] 고시 수집 및 Qdrant+RDB 적재 시작...")
            rdb = run_usage_standard_pipeline(
                collection_name=collection,
                skip_qdrant=skip_vector_db,
                database_url=db_url,
                # PDF 단계에서 execute_payload_to_rdb가 이미 profiles를 적재했으면 중복 방지
                skip_profiles=bool(pdf_files and db_url),
            )
            usage_std_qdrant = rdb["qdrant"]
            usage_std_master = rdb["master"]
            usage_std_corpus = rdb["corpus"]
            usage_std_rules = rdb["rules"]
            usage_std_profiles = rdb["profiles"]
            print(
                f"  [산안비 사용기준] 완료: "
                f"Qdrant {usage_std_qdrant}개, "
                f"Master {usage_std_master}개 (Corpus {usage_std_corpus} / Rules {usage_std_rules}), "
                f"Profiles {usage_std_profiles}개"
            )
        except Exception as e:
            log.warning("산안비 사용기준 파이프라인 실패 (건너뜀): %s", e)

    result = {
        "pdf_chunks": len(pdf_chunks),  # build_payload 기반 Qdrant 문서 수
        "total_chunks": len(pdf_chunks) + law_api_chunks + usage_std_qdrant,
        "law_api_chunks": law_api_chunks,
        # PDF → RDB
        "pdf_rdb_master": pdf_rdb_master,
        "pdf_rdb_corpus": pdf_rdb_corpus,
        "pdf_rdb_rules": pdf_rdb_rules,
        "pdf_rdb_profiles": pdf_rdb_profiles,
        # 법제처 Open API → RDB
        "law_api_rdb": law_api_rdb,
        # 산안비 사용기준 → Qdrant + RDB
        "usage_standard_qdrant": usage_std_qdrant,
        "usage_standard_master": usage_std_master,
        "usage_standard_corpus": usage_std_corpus,
        "usage_standard_rules": usage_std_rules,
        "usage_standard_profiles": usage_std_profiles,
        "skipped": False,
    }
    _save_state(collection, result)
    return result


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
