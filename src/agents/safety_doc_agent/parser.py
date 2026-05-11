from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# "**1. 개요**", "**3-2. 보호구**" 같은 제목 패턴을 잡는다.
# 일부 마크다운 내보내기 결과는 "1\."처럼 이스케이프된 점을 쓰기 때문에
# 그 경우도 함께 허용한다.
SECTION_RE = re.compile(r"^\*\*(\d+(?:-\d+)?)(?:\\)?\.\s*(.+?)\*\*$")


@dataclass(slots=True)
class GuideChunk:
    """원본 가이드에서 추출한 검색 친화적 청크."""

    chunk_id: str
    section_key: str
    section_title: str
    chunk_type: str
    text: str
    metadata: dict[str, Any]


def _clean_line(line: str) -> str:
    """저장용 청크가 일반 문장처럼 읽히도록 마크다운 목록 기호를 제거한다."""

    line = line.strip()
    line = re.sub(r"^\*+\s*", "", line)
    return line.strip()


def _split_docs(cell: str) -> list[str]:
    """표 셀 안에 여러 증빙이 함께 적힌 경우 항목별로 분리한다."""

    parts = [part.strip() for part in re.split(r",|\n", cell) if part.strip()]
    return [re.sub(r"\s+", " ", part) for part in parts]


def parse_markdown_table(markdown: str) -> list[dict[str, str]]:
    """첫 번째 마크다운 표를 행 단위 딕셔너리로 추출한다.

    원본 가이드에는 핵심 체크리스트 표가 하나 있고, 이 표를 이후
    벡터 검색과 규칙 기반 매칭 양쪽에서 모두 활용한다.
    """

    lines = markdown.splitlines()
    table_lines: list[str] = []
    in_table = False

    for line in lines:
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
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))

    return rows


def parse_guide(guide_path: str | Path) -> dict[str, Any]:
    """안전 가이드를 구조화된 섹션과 검색용 청크로 파싱한다."""

    path = Path(guide_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        match = SECTION_RE.match(line.strip())
        if match:
            if current:
                sections.append(current)
            current = {
                "key": match.group(1),
                "title": match.group(2).strip(),
                "lines": [],
            }
            continue

        if current is not None:
            current["lines"].append(line.rstrip())

    if current:
        sections.append(current)

    # 체크리스트 표는 자유 서술보다 바로 쓰기 쉬운 구조이므로 별도로 파싱해
    # 정규화된 requirement 항목으로 변환한다.
    evidence_rows = parse_markdown_table(text)

    parsed_sections: list[dict[str, Any]] = []
    chunks: list[GuideChunk] = []

    for section in sections:
        cleaned_lines = [_clean_line(line) for line in section["lines"] if _clean_line(line)]
        body = "\n".join(cleaned_lines).strip()
        parsed_sections.append(
            {
                "key": section["key"],
                "title": section["title"],
                "body": body,
            }
        )

        if body:
            chunks.append(
                GuideChunk(
                    chunk_id=f"section-{section['key']}",
                    section_key=section["key"],
                    section_title=section["title"],
                    chunk_type="section",
                    text=body,
                    metadata={
                        "section_key": section["key"],
                        "section_title": section["title"],
                    },
                )
            )

    checklist: list[dict[str, Any]] = []
    for index, row in enumerate(evidence_rows, start=1):
        category = row.get("항목 구분", "").strip()
        docs_raw = row.get("주요 증빙 및 필수 서류", "").strip()
        tips = row.get("실무 포인트", "").strip()
        documents = _split_docs(docs_raw)
        item = {
            "category": category,
            "required_documents": documents,
            "practical_point": tips,
        }
        checklist.append(item)
        # 체크리스트 행을 별도 청크로 만들어야 검색 시 넓은 섹션 설명이 아니라
        # 실제 증빙 요구사항을 더 정확하게 끌어올 수 있다.
        chunks.append(
            GuideChunk(
                chunk_id=f"checklist-{index}",
                section_key="4",
                section_title="주요 증빙 서류 및 청구 원칙",
                chunk_type="checklist",
                text=f"{category}\n필수서류: {', '.join(documents)}\n실무포인트: {tips}",
                metadata={
                    "category": category,
                    "required_documents": json.dumps(documents, ensure_ascii=False),
                    "practical_point": tips,
                },
            )
        )

    return {
        "source_path": str(path),
        "sections": parsed_sections,
        "checklist": checklist,
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def save_parsed_guide(parsed: dict[str, Any], output_path: str | Path) -> None:
    """점검 명령에서 재사용할 수 있도록 파싱 결과를 파일로 저장한다."""

    Path(output_path).write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_parsed_guide(path: str | Path) -> dict[str, Any]:
    """이전에 저장한 파싱 결과를 다시 불러온다."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
