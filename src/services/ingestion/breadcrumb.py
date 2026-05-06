import re


def _build_breadcrumb(heading_stack: dict, current_level: int) -> str:
    crumbs = [heading_stack[l] for l in sorted(heading_stack) if l < current_level]
    return " > ".join(crumbs)


def inject_breadcrumbs(markdown_text: str) -> str:
    """
    ### 이하 헤딩 바로 아래에 HTML 주석으로 상위 섹션 경로를 삽입.

        ### 나 계상 시기
        <!-- context: 01 해설집 > Ⅱ 산업안전보건관리비의 계상 등 -->
    """
    heading_stack: dict[int, str] = {}
    output_lines = []

    for line in markdown_text.splitlines(keepends=True):
        m = re.match(r"^(#+) (.+)", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()

            for l in list(heading_stack):
                if l >= level:
                    del heading_stack[l]
            heading_stack[level] = text

            output_lines.append(line)

            if level >= 3:
                breadcrumb = _build_breadcrumb(heading_stack, level)
                if breadcrumb:
                    output_lines.append(f"<!-- context: {breadcrumb} -->\n")
        else:
            output_lines.append(line)

    return "".join(output_lines)
