from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.storage import DEFAULT_EMBED_MODEL, load_vectorstore


SECTION_RE = re.compile(r"^\*\*(\d+(?:-\d+)?)(?:\\)?\.\s*(.+?)\*\*$")


@dataclass(slots=True)
class ReferenceChunk:
    chunk_id: str
    section_key: str
    section_title: str
    chunk_type: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class VectorSettings:
    openai_api_key: str
    embedding_model: str
    qdrant_path: str
    qdrant_url: str
    qdrant_api_key: str


def load_settings() -> VectorSettings:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    return VectorSettings(
        openai_api_key=api_key,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        qdrant_path=os.getenv("SAFETY_DOC_QDRANT_PATH", ".qdrant/safety-doc-reference").strip(),
        qdrant_url=os.getenv("SAFETY_DOC_QDRANT_URL", os.getenv("QDRANT_URL", "")).strip(),
        qdrant_api_key=os.getenv("SAFETY_DOC_QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", "")).strip(),
    )


def parse_reference_markdown(source_path: str | Path) -> dict[str, Any]:
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        match = SECTION_RE.match(line.strip())
        if match:
            if current:
                sections.append(current)
            current = {"key": match.group(1), "title": match.group(2).strip(), "lines": []}
            continue
        if current is not None:
            current["lines"].append(line.rstrip())

    if current:
        sections.append(current)

    chunks: list[ReferenceChunk] = []
    parsed_sections: list[dict[str, str]] = []
    for section in sections:
        body = "\n".join(_clean_line(line) for line in section["lines"] if _clean_line(line)).strip()
        parsed_sections.append({"key": section["key"], "title": section["title"], "body": body})
        if body:
            chunks.append(
                ReferenceChunk(
                    chunk_id=f"section-{section['key']}",
                    section_key=section["key"],
                    section_title=section["title"],
                    chunk_type="section",
                    text=body,
                    metadata={"source_path": str(path), "section_key": section["key"], "section_title": section["title"]},
                )
            )

    for index, row in enumerate(_parse_first_markdown_table(text), start=1):
        category = row.get("항목 구분", "").strip()
        docs = _split_docs(row.get("주요 증빙 및 필수 서류", ""))
        practical_point = row.get("실무 포인트", "").strip()
        chunks.append(
            ReferenceChunk(
                chunk_id=f"checklist-{index}",
                section_key="checklist",
                section_title="주요 증빙 서류 및 청구 원칙",
                chunk_type="checklist",
                text=f"{category}\n필수서류: {', '.join(docs)}\n실무포인트: {practical_point}",
                metadata={
                    "source_path": str(path),
                    "category": category,
                    "required_documents": json.dumps(docs, ensure_ascii=False),
                    "practical_point": practical_point,
                },
            )
        )

    return {"source_path": str(path), "sections": parsed_sections, "chunks": [asdict(chunk) for chunk in chunks]}


def build_reference_vector_db(source_path: str | Path, collection_name: str) -> int:
    settings = load_settings()
    parsed = parse_reference_markdown(source_path)
    chunks = parsed["chunks"]
    if not chunks:
        raise ValueError(f"No reference chunks extracted from {source_path}")

    openai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant = _get_qdrant_client(settings)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed_texts(openai_client, settings.embedding_model, texts)
    _ensure_collection(qdrant, collection_name, len(embeddings[0]))

    points = [
        PointStruct(
            id=_point_id(str(chunk["chunk_id"])),
            vector=embedding,
            payload={
                "chunk_id": chunk["chunk_id"],
                "document": chunk["text"],
                "chunk_type": chunk["chunk_type"],
                **chunk["metadata"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_reference_vector_db(query: str, collection_name: str, top_k: int) -> list[dict[str, Any]]:
    embed_model = os.getenv("SAFETY_DOC_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL).strip()
    vectorstore = load_vectorstore(
        collection_name,
        qdrant_url=os.getenv("SAFETY_DOC_QDRANT_URL", os.getenv("QDRANT_URL", "")).strip() or None,
        embed_model=embed_model,
    )
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {
            "id": str(document.metadata.get("_id") or document.metadata.get("id") or ""),
            "score": score,
            "payload": {
                "text": document.page_content,
                "source": document.metadata.get("source"),
                "section": document.metadata.get("section_title"),
                "metadata": document.metadata,
            },
        }
        for document, score in results
    ]


def _clean_line(line: str) -> str:
    return re.sub(r"^\*+\s*", "", line.strip()).strip()


def _split_docs(cell: str) -> list[str]:
    return [re.sub(r"\s+", " ", part.strip()) for part in re.split(r",|\n", cell) if part.strip()]


def _parse_first_markdown_table(markdown: str) -> list[dict[str, str]]:
    table_lines: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            in_table = True
            table_lines.append(line.strip())
        elif in_table:
            break

    if len(table_lines) < 3:
        return []

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for raw_row in table_lines[2:]:
        cells = [cell.strip() for cell in raw_row.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def _get_qdrant_client(settings: VectorSettings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def _embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"safety-doc-reference:{chunk_id}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="safety-doc-agent 참고자료용 Qdrant 벡터DB 도구")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_ = subparsers.add_parser("build", help="마크다운 참고자료를 청크로 추출해 Qdrant에 적재합니다.")
    build_parser_.add_argument("--source", required=True, help="참고자료 마크다운 경로")
    build_parser_.add_argument("--collection", required=True, help="Qdrant 컬렉션명")
    build_parser_.set_defaults(func=cmd_build)

    search_parser = subparsers.add_parser("search", help="참고자료 Qdrant 컬렉션에서 관련 청크를 검색합니다.")
    search_parser.add_argument("--query", required=True, help="검색 질의")
    search_parser.add_argument("--collection", required=True, help="Qdrant 컬렉션명")
    search_parser.add_argument("--top-k", type=int, default=6, help="검색 결과 개수")
    search_parser.set_defaults(func=cmd_search)

    return parser


def cmd_build(args: argparse.Namespace) -> None:
    count = build_reference_vector_db(args.source, args.collection)
    print(json.dumps({"collection": args.collection, "chunks_indexed": count}, ensure_ascii=False, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    results = search_reference_vector_db(args.query, args.collection, args.top_k)
    print(json.dumps({"collection": args.collection, "results": results}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
