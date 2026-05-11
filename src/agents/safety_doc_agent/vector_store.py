from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Settings


def get_openai_client(settings: Settings) -> OpenAI:
    """검증된 설정으로 OpenAI 클라이언트를 만들고 필요 시 LangSmith를 연결한다."""

    client = OpenAI(api_key=settings.openai_api_key)
    if settings.langsmith_tracing:
        return wrap_openai(client)
    return client


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """원격 Qdrant와 로컬 디스크 기반 Qdrant를 모두 지원한다."""

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """저장과 검색이 같은 모델을 쓰도록 텍스트 배치를 임베딩한다."""

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def to_qdrant_point_id(chunk_id: str) -> str:
    """사람이 읽기 쉬운 청크 ID를 Qdrant용 결정적 UUID로 바꾼다.

    로컬 Qdrant는 포인트 ID 형식에 엄격하므로, 원래 청크 ID는 payload에
    보관하고 저장용 ID만 안정적인 UUID로 변환한다.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"safety-doc-agent:{chunk_id}"))


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """첫 인덱싱 시점에 대상 컬렉션이 없으면 생성한다."""

    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def index_chunks(parsed: dict[str, Any], settings: Settings, collection_name: str) -> int:
    """파싱된 가이드 청크를 OpenAI 임베딩과 함께 Qdrant에 저장한다."""

    qdrant = get_qdrant_client(settings)
    openai_client = get_openai_client(settings)

    chunks = parsed["chunks"]
    ids = [chunk["chunk_id"] for chunk in chunks]
    docs = [chunk["text"] for chunk in chunks]
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        # Qdrant payload는 원시 타입 위주가 다루기 쉬우므로, 중첩 객체 대신
        # 여기서 평탄화하고 문자열화한 메타데이터만 저장한다.
        metadata = {
            "section_key": str(chunk["metadata"].get("section_key", chunk["section_key"])),
            "section_title": str(chunk["metadata"].get("section_title", chunk["section_title"])),
            "chunk_type": str(chunk["chunk_type"]),
        }
        for key, value in chunk["metadata"].items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        metadatas.append(metadata)

    embeddings = embed_texts(openai_client, settings.embedding_model, docs)
    ensure_collection(qdrant, collection_name, vector_size=len(embeddings[0]))

    points = [
        PointStruct(
            id=to_qdrant_point_id(chunk_id),
            vector=embedding,
            payload={
                "chunk_id": chunk_id,
                "document": document,
                **metadata,
            },
        )
        for chunk_id, embedding, document, metadata in zip(ids, embeddings, docs, metadatas)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
    return len(ids)


def retrieve_relevant_chunks(
    query: str,
    settings: Settings,
    collection_name: str,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """현재 점검 질의와 가장 관련 있는 가이드 청크를 검색한다."""

    qdrant = get_qdrant_client(settings)
    client = get_openai_client(settings)
    query_embedding = embed_texts(client, settings.embedding_model, [query])[0]
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    chunks: list[dict[str, Any]] = []
    for point in results.points:
        payload = point.payload or {}
        # "document"는 프롬프트에서 바로 쓰기 쉽도록 분리하고, 나머지는
        # 점검 근거와 디버깅용 메타데이터로 유지한다.
        metadata = {key: value for key, value in payload.items() if key != "document"}
        chunks.append(
            {
                "id": str(payload.get("chunk_id", point.id)),
                "document": str(payload.get("document", "")),
                "metadata": metadata,
            }
        )
    return chunks
