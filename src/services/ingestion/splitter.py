# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 함수 정의 ]
#
# 1. split_markdown()        : 마크다운 법령 문서 → LangChain Document 청크 분할
# 2. _merge_small_chunks()   : 소형 청크 병합으로 품질 향상
# 3. _split_large_chunk()    : 대형 청크 재귀 분할
# 4. _should_drop_chunk()    : 불필요 청크(목차, 서식 헤더 등) 제거 판별
# --------------------------------------------------------------------------
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS_TO_SPLIT = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]

# 한국 법령 조항 패턴: 제X조, 제X조제Y항, 제X조제Y항제Z호, ...목
_LAW_ARTICLE_RE = re.compile(
    r"제\d+조(?:의\d+)?(?:제\d+항(?:제\d+호(?:[가-하]목)?)?)?"
)
_EXISTING_CITE_RE = re.compile(r"\[LEGAL_CITE:\s*([^\]]+)\]")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")
_QNA_BOUNDARY_RE = re.compile(r"(?m)(?=^####\s|\n-\s*\d+\)\s|\n\d+\)\s)")
_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)

_TARGET_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 120
_MAX_CHUNK_SIZE = 1800
_MIN_CHUNK_SIZE = 150
_HEADING_ONLY_MAX = 120


def _extract_breadcrumb(content: str) -> str | None:
    m = re.search(r"<!--\s*context:\s*(.+?)\s*-->", content)
    return m.group(1) if m else None


def _inject_article_cites(doc) -> None:
    """본문에 등장하는 법령 조항 번호를 LEGAL_CITE 태그로 자동 주입."""
    existing = set()
    for raw in _EXISTING_CITE_RE.findall(doc.page_content):
        for part in raw.split("|"):
            existing.add(part.strip())

    found = set(_LAW_ARTICLE_RE.findall(doc.page_content)) - existing
    if found:
        cite_str = " | ".join(sorted(found))
        doc.page_content = f"[LEGAL_CITE: {cite_str}]\n" + doc.page_content


def _clean_chunk(doc) -> None:
    breadcrumb = _extract_breadcrumb(doc.page_content)
    if breadcrumb:
        doc.metadata["breadcrumb"] = breadcrumb

    doc.page_content = re.sub(r"<!--\s*context:.*?-->\n?", "", doc.page_content)
    doc.page_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)
    doc.page_content = re.sub(r"﻿", "", doc.page_content)
    doc.page_content = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", doc.page_content)
    doc.page_content = doc.page_content.strip()

    _inject_article_cites(doc)


def _clone_doc(doc: Document, page_content: str) -> Document:
    return Document(page_content=page_content, metadata=dict(doc.metadata))


def _strip_leading_cites(text: str) -> str:
    lines = text.splitlines()
    while lines and lines[0].startswith("[LEGAL_CITE:"):
        lines.pop(0)
    return "\n".join(lines).strip()


def _is_heading_only_chunk(text: str) -> bool:
    body = _strip_leading_cites(text)
    if not body:
        return True

    non_empty = [line.strip() for line in body.splitlines() if line.strip()]
    if not non_empty:
        return True

    if len(body) > _HEADING_ONLY_MAX:
        return False

    if len(non_empty) == 1 and (
        _MARKDOWN_HEADING_RE.match(non_empty[0]) or len(non_empty[0]) <= 40
    ):
        return True

    heading_like = sum(1 for line in non_empty if _MARKDOWN_HEADING_RE.match(line))
    return heading_like == len(non_empty)


def _split_large_chunk(doc: Document) -> list[Document]:
    text = doc.page_content.strip()
    if len(text) <= _MAX_CHUNK_SIZE:
        return [doc]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_TARGET_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=[
            "\n\n#### ",
            "\n\n### ",
            "\n\n## ",
            "\n\n",
            "\n- ",
            "\n• ",
            "\n| ",
            "\n",
            " ",
            "",
        ],
    )

    segments: list[str] = []
    qna_parts = [part.strip() for part in _QNA_BOUNDARY_RE.split(text) if part.strip()]
    if len(qna_parts) > 1:
        segments.extend(qna_parts)
    else:
        segments.append(text)

    split_docs: list[Document] = []
    for segment in segments:
        if len(segment) <= _MAX_CHUNK_SIZE:
            split_docs.append(_clone_doc(doc, segment))
            continue

        if _TABLE_LINE_RE.search(segment):
            table_parts = [part.strip() for part in re.split(r"(?m)(?=^\|)", segment) if part.strip()]
            if len(table_parts) > 1:
                for table_part in table_parts:
                    if len(table_part) <= _MAX_CHUNK_SIZE:
                        split_docs.append(_clone_doc(doc, table_part))
                    else:
                        for piece in splitter.split_text(table_part):
                            split_docs.append(_clone_doc(doc, piece))
                continue

        for piece in splitter.split_text(segment):
            split_docs.append(_clone_doc(doc, piece))

    return split_docs


def _merge_small_chunks(chunks: list[Document]) -> list[Document]:
    if not chunks:
        return []

    merged: list[Document] = []
    idx = 0
    while idx < len(chunks):
        current = chunks[idx]
        text = current.page_content.strip()
        if not text:
            idx += 1
            continue

        is_small = len(text) < _MIN_CHUNK_SIZE or _is_heading_only_chunk(text)
        if is_small and idx + 1 < len(chunks):
            next_doc = chunks[idx + 1]
            combined = f"{text}\n\n{next_doc.page_content.strip()}".strip()
            merged.append(_clone_doc(next_doc, combined))
            idx += 2
            continue

        if merged and len(text) < _MIN_CHUNK_SIZE:
            prev = merged.pop()
            combined = f"{prev.page_content.strip()}\n\n{text}".strip()
            merged.append(_clone_doc(prev, combined))
            idx += 1
            continue

        merged.append(current)
        idx += 1

    return merged


def _looks_like_table_of_contents(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if "CONTENTS" in compact.upper():
        return True
    if "···" in compact and re.search(r"\|\s*\d+\s*\|?$", compact):
        return True
    return False


def _is_form_header(metadata: dict) -> bool:
    for key in ("header_1", "header_2", "header_3", "header_4", "breadcrumb"):
        value = str(metadata.get(key, ""))
        if "【별지" in value or "서식" in value:
            return True
    return False


def _should_drop_chunk(doc: Document) -> bool:
    text = doc.page_content.strip()
    metadata = doc.metadata or {}

    if not text:
        return True

    if _looks_like_table_of_contents(text):
        return True

    if _is_form_header(metadata) and len(text) < 300 and "[LEGAL_CITE:" not in text:
        return True

    if _TABLE_LINE_RE.fullmatch(text) and len(text) < 250:
        return True

    return False


def split_markdown(markdown_text: str, source_metadata: dict | None = None) -> list:
    """
    최종 마크다운을 헤더 기준으로 청킹하고, 출처 메타데이터를 각 청크에 주입.

    source_metadata 예시:
        {"source": "파일명.pdf", "source_stem": "파일명"}
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT, strip_headers=False
    )
    raw_chunks = splitter.split_text(markdown_text)

    chunks: list[Document] = []
    for doc in raw_chunks:
        _clean_chunk(doc)
        if source_metadata:
            doc.metadata.update(source_metadata)
        if doc.page_content:
            chunks.extend(_split_large_chunk(doc))

    merged = _merge_small_chunks(chunks)
    final_chunks: list[Document] = []
    for doc in merged:
        _clean_chunk(doc)
        if doc.page_content and not _should_drop_chunk(doc):
            final_chunks.append(doc)

    return final_chunks
