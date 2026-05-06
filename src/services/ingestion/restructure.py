import re


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _classify_heading(text: str) -> int:
    t = _clean_text(text)

    if re.match(r"^건설업\s+산업안전\s*보건관리비\s+해설", t):
        return 1
    if re.match(r"^건설업\s+산업안전보건관리비$", t):
        return 1

    if re.match(r"^0[0-9]\s", t):
        return 2
    if re.match(r"^제\d+장\s", t):
        return 2

    if re.match(r"^[※*]", t):
        return 5

    if re.match(r"^제\d+조", t):
        return 3

    if re.match(r"^\d+\)", t):
        return 4

    return 3


# 🔥 핵심: 다중 법령 추출
def _extract_all_legal_cites(text: str):
    """
    한 문장에서 모든 법령 조합 추출
    ex) 제36조 제3호, 제4조 제3호, 제24조 ...
    """

    pattern = r"(제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?)"
    matches = re.findall(pattern, text)

    cleaned = []
    for m in matches:
        m = re.sub(r"\s+", "", m)
        cleaned.append(m)

    # 중복 제거 + 순서 유지
    seen = set()
    result = []
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


# 🔥 리스트 구조 판단
def _is_list_line(text: str) -> bool:
    return bool(
        re.match(r"^\s*[-•▪]", text)
        or re.match(r"^\s*\d+\.", text)
        or re.match(r"^\s*\d+\)", text)
        or re.match(r"^\s*[가-힣]\.", text)
        or re.match(r"^\s*[가-힣]\)", text)
    )


def restructure_markdown(markdown_text: str) -> str:

    markdown_text = re.sub(r"<!--\s*image\s*-->\n?", "", markdown_text)

    output_lines = []

    # 🔥 carry 상태
    current_article = None
    current_paragraph = None
    carry_depth = 0
    MAX_CARRY = 5

    for line in markdown_text.splitlines(keepends=True):
        # ── 헤딩 ──
        if line.startswith("## "):
            cleaned = _clean_text(line[3:])
            level = _classify_heading(cleaned)

            article = re.search(r"제\s*(\d+)조", cleaned)
            paragraph = re.search(r"제\s*(\d+)항", cleaned)

            if article:
                current_article = f"제{article.group(1)}조"
                current_paragraph = None
                carry_depth = 0

            if paragraph:
                current_paragraph = f"제{paragraph.group(1)}항"
                carry_depth = 0

            output_lines.append(f"{'#' * level} {cleaned}\n")
            continue

        # ── 빈 줄 ──
        if not line.strip():
            output_lines.append(line)
            carry_depth += 1
            continue

        # 🔥 1순위: 문장 내 다중 법령 추출
        cites = _extract_all_legal_cites(line)

        if cites:
            output_lines.append(f"[LEGAL_CITE: {' | '.join(cites)}] {line}")

            # carry 업데이트 (첫 번째 기준)
            first = cites[0]

            article = re.search(r"제\d+조", first)
            paragraph = re.search(r"제\d+항", first)

            if article:
                current_article = article.group()
            if paragraph:
                current_paragraph = paragraph.group()

            carry_depth = 0
            continue

        # 🔥 2순위: 리스트 기반 carry
        if _is_list_line(line) and carry_depth < MAX_CARRY:
            parts = []
            if current_article:
                parts.append(current_article)
            if current_paragraph:
                parts.append(current_paragraph)

            if parts:
                output_lines.append(f"[LEGAL_CITE: {' '.join(parts)}] {line}")
                carry_depth += 1
                continue

        # 🔥 3순위: 일반 문장
        carry_depth += 1
        if carry_depth > MAX_CARRY:
            current_article = None
            current_paragraph = None

        output_lines.append(line)

    return "".join(output_lines)
