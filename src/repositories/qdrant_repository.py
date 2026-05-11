"""
Qdrant 벡터 DB 클라이언트 래퍼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — Vector Store 파트

[역할]
  매칭 엔진에서 사용하는 텍스트(품목명, 법령 조항 등)를
  벡터로 변환 후 Qdrant에 저장·검색한다.

[컬렉션 이름 정책]
  collection_name은 .env에 고정하지 않고
  각 사용처(클래스 생성 시 또는 메서드 호출 시)에서 직접 전달한다.

  이유:
    · 서비스 파트(OCR, RAG 등)마다 컬렉션 구조가 달라
      하나의 환경변수로 통일하기 어려움
    · 테스트용 컬렉션을 독립적으로 운영할 때 유연하게 지정 가능

  사용 예:
    repo = QdrantRepository(collection_name="safety_items")
    repo.upsert(points)
    results = repo.search(query_vector, top_k=5)

[연결 설정]
  QDRANT_URL     (필수) — Qdrant 서버 주소 (기본: http://localhost:6333)
  QDRANT_API_KEY (선택) — Qdrant Cloud 사용 시 API 키
  두 값 모두 .env 또는 환경변수에서 읽는다.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
    Filter,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 기본 상수
# ──────────────────────────────────────────────────────────────────────

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_VECTOR_SIZE = 1536        # OpenAI text-embedding-3-small 기준
DEFAULT_DISTANCE = Distance.COSINE


# ──────────────────────────────────────────────────────────────────────
# 클라이언트 싱글톤 팩토리
# ──────────────────────────────────────────────────────────────────────

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    QdrantClient 싱글톤 반환.

    환경변수:
        QDRANT_URL      — Qdrant 서버 주소 (기본: http://localhost:6333)
        QDRANT_API_KEY  — Qdrant Cloud 사용 시 API 키 (로컬 실행 시 불필요)
    """
    global _client
    if _client is None:
        url     = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        api_key = os.getenv("QDRANT_API_KEY")  # 로컬 Qdrant는 None으로 무방
        _client = QdrantClient(url=url, api_key=api_key)
        logger.info("Qdrant 클라이언트 연결: %s", url)
    return _client


# ──────────────────────────────────────────────────────────────────────
# Repository 클래스
# ──────────────────────────────────────────────────────────────────────

class QdrantRepository:
    """
    Qdrant 컬렉션 단위 CRUD 래퍼.

    Args:
        collection_name : 사용할 컬렉션 이름.
                          .env 고정이 아닌 호출 시점에 전달.
                          예) "safety_items", "law_articles", "usage_history"
        vector_size     : 임베딩 벡터 차원 수 (기본: 1536, OpenAI 기준)
        distance        : 유사도 계산 방식 (기본: COSINE)
        client          : 외부에서 주입할 클라이언트 (미전달 시 싱글톤 사용)
    """

    def __init__(
        self,
        collection_name: str,
        vector_size:     int             = DEFAULT_VECTOR_SIZE,
        distance:        Distance        = DEFAULT_DISTANCE,
        client:          Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.vector_size     = vector_size
        self.distance        = distance
        self._client         = client or get_qdrant_client()

    # ── 컬렉션 관리 ──────────────────────────────────────────────────

    def create_collection_if_not_exists(self) -> bool:
        """
        컬렉션이 없을 때만 생성한다.

        Returns:
            True  — 새로 생성됨
            False — 이미 존재
        """
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            logger.debug("컬렉션 이미 존재: %s", self.collection_name)
            return False

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )
        logger.info("컬렉션 생성: %s (size=%d, distance=%s)",
                    self.collection_name, self.vector_size, self.distance)
        return True

    def delete_collection(self) -> None:
        """컬렉션 삭제 (데이터 포함)."""
        self._client.delete_collection(self.collection_name)
        logger.info("컬렉션 삭제: %s", self.collection_name)

    def collection_info(self) -> dict:
        """컬렉션 상태 정보 반환."""
        info = self._client.get_collection(self.collection_name)
        return {
            "name":         self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count":  info.points_count,
            "status":        str(info.status),
        }

    # ── 데이터 적재 ──────────────────────────────────────────────────

    def upsert(self, points: list[PointStruct]) -> None:
        """
        벡터 포인트 일괄 적재 (insert or update).

        Args:
            points : PointStruct 리스트.
                     id        — 문서 고유 ID (int 또는 UUID 문자열)
                     vector    — 임베딩 벡터 (float list)
                     payload   — 메타데이터 dict (품목명, 날짜, 금액 등)

        사용 예:
            from qdrant_client.http.models import PointStruct
            repo.upsert([
                PointStruct(
                    id=1,
                    vector=[0.1, 0.2, ...],
                    payload={"item_name": "안전모", "amount": 150000},
                )
            ])
        """
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.debug("%d개 포인트 upsert → %s", len(points), self.collection_name)

    # ── 검색 ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k:        int                 = 5,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter]   = None,
    ) -> list[ScoredPoint]:
        """
        유사도 검색.

        Args:
            query_vector    : 검색 쿼리 임베딩 벡터
            top_k           : 반환할 최대 결과 수 (기본: 5)
            score_threshold : 이 값 이상의 유사도 결과만 반환 (None = 전체)
            query_filter    : Qdrant 필터 조건 (payload 기준 필터링)

        Returns:
            ScoredPoint 리스트 (score 내림차순 정렬)
        """
        results = self._client.search(
            collection_name  = self.collection_name,
            query_vector     = query_vector,
            limit            = top_k,
            score_threshold  = score_threshold,
            query_filter     = query_filter,
            with_payload     = True,
        )
        logger.debug("검색 결과 %d건 ← %s", len(results), self.collection_name)
        return results

    def search_as_dicts(
        self,
        query_vector: list[float],
        top_k:        int                 = 5,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter]   = None,
    ) -> list[dict]:
        """
        search() 결과를 dict 리스트로 변환해 반환.
        FastAPI 응답이나 매칭 엔진 입력에 바로 쓸 수 있는 형태.

        Returns:
            [{"id": ..., "score": ..., "payload": {...}}, ...]
        """
        return [
            {
                "id":      hit.id,
                "score":   hit.score,
                "payload": hit.payload or {},
            }
            for hit in self.search(query_vector, top_k, score_threshold, query_filter)
        ]

    # ── 단건 조회 / 삭제 ─────────────────────────────────────────────

    def get_by_id(self, point_id: int | str) -> Optional[dict]:
        """ID로 단건 조회. 없으면 None."""
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        p = results[0]
        return {"id": p.id, "payload": p.payload or {}}

    def delete_by_ids(self, point_ids: list[int | str]) -> None:
        """ID 목록으로 포인트 삭제."""
        from qdrant_client.http.models import PointIdsList
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.debug("%d개 포인트 삭제 ← %s", len(point_ids), self.collection_name)
