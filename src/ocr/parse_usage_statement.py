"""
산업안전보건관리비 사용내역서 PDF 파싱 모듈 v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 공식 서식(별지 제1호, 10페이지) 및 단순 서식 자동 감지
- pdfplumber 기반 테이블 파싱
- JSON 출력 → usage_statement_items DB 테이블 구조에 맞춘 표준 출력

사용법:
    python parse_usage_statement.py --pdf 사용내역서.pdf
    python parse_usage_statement.py --folder ./pdfs/ --output ./results/

설치:
    pip install pdfplumber
"""

import argparse
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import pdfplumber

# ══════════════════════════════════════════════════════════
# 0. 상수 정의
# ══════════════════════════════════════════════════════════

# ── 사용내역서 판별 키워드 ────────────────────────────────
# 키워드 중 MIN_KEYWORD_MATCH개 이상 발견되면 사용내역서로 판별
# 실제 사용내역서를 확인하면서 키워드 추가/제거 가능
USAGE_STATEMENT_KEYWORDS = [
    "사용내역서",
    "산업안전보건관리비",
    "소재지",
    "대표자",
    "공사금액",
    "공사기간",
    "발주자",
    "공정률",
    "공정율",
    "안전관리비",
]
MIN_KEYWORD_MATCH = 6  # 최소 매칭 키워드 수


# 공식 서식(별지 제1호): 페이지 번호 → DB 카테고리 코드 (V4__seed_types.sql 기준)
PAGE_CATEGORY_MAP = {
    1: None,  # 요약 페이지
    2: "CAT_01",  # 안전·보건관리자 임금 등
    3: "CAT_02",  # 안전시설비 등
    4: "CAT_03",  # 보호구 등
    5: "CAT_04",  # 안전보건진단비 등
    6: "CAT_05",  # 안전보건교육비 등
    7: "CAT_06",  # 근로자 건강장해예방비 등
    8: "CAT_07",  # 건설재해예방전문지도기관 기술지도비
    9: "CAT_08",  # 본사 전담조직 근로자 임금 등
    10: "CAT_09",  # 위험성평가 등에 따른 소요비용
}

# 페이지 번호별 스킵 사유 (None 으로 처리되는 페이지 중 경고가 필요한 경우)
PAGE_SKIP_REASONS: dict[int, str] = {}  # 현재 스킵 대상 없음 (요약 페이지는 정상 처리)

CATEGORY_NAME_MAP = {
    "CAT_01": "안전·보건관리자 임금 등",
    "CAT_02": "안전시설비 등",
    "CAT_03": "보호구 등",
    "CAT_04": "안전보건진단비 등",
    "CAT_05": "안전보건교육비 등",
    "CAT_06": "근로자 건강장해예방비 등",
    "CAT_07": "건설재해예방전문지도기관 기술지도비",
    "CAT_08": "본사 전담조직 근로자 임금 등",
    "CAT_09": "위험성평가 등에 따른 소요비용",
}

# 카테고리별 컬럼 구조 정의 (DB 카테고리 코드 기준 — CAT_01~CAT_09)
CATEGORY_COLUMNS = {
    "CAT_01": {
        "date": ["지급일", "지급 일"],
        "description": ["지급 내역", "지급내역", "내역"],
        "amount": ["지급금액", "금액"],
        "extra": {
            "name": ["성 명", "성명"],
            "org": ["소 속", "소속"],
            "appointed": ["선임일"],
        },
    },
    "CAT_02": {
        "date": ["사용일", "사용일자"],
        "description": ["지급내역", "지급 내역", "사용내역", "사용 내역", "사 용 내 역"],
        "amount": ["사용금액", "금액", "계"],
        "extra": {
            "unit": ["단위"],
            "qty": ["수량"],
            "labor": ["노무비"],
            "material": ["자재비"],
        },
    },
    "CAT_03": {
        "date": ["사용일", "사용일자"],
        "description": ["품 목", "품목", "지급내역", "지급 내역", "사용내역", "사용 내역", "사 용 내 역"],
        "amount": ["금액", "소요 비용", "소요비용"],
        "extra": {
            "unit_price": ["단가"],
            "qty": ["수량"],
        },
    },
    "CAT_04": {
        "date": ["사용일", "사용일자"],
        "description": ["사 용 내 역", "사용내역", "사용 내역", "품 목", "품목"],
        "amount": ["소요비용", "소요 비용", "금액"],
        "extra": {
            "institution": ["진단기관", "검사기관"],
        },
    },
    # CAT_05: 안전보건교육비 등 (공식 서식 6페이지)
    "CAT_05": {
        "date": ["교육일", "교육일자", "사용일", "사용일자"],
        "description": ["교육 내용", "교육내용", "교육과목", "사 용 내 역", "사용내역", "사용 내역"],
        "amount": ["소요 경비", "소요경비", "금액"],
        "extra": {
            "institution": ["교육기관"],
            "participants": ["교육인원", "인원"],
        },
    },
    # CAT_06: 근로자 건강장해예방비 등 (공식 서식 7페이지)
    "CAT_06": {
        "date": ["사용일", "사용일자"],
        "description": ["사 용 내 역", "사용내역", "사용 내역", "품 목", "품목"],
        "amount": ["소요 경비", "소요경비", "금액"],
        "extra": {
            "hospital": ["진단병원"],
            "participants": ["참가인원"],
        },
    },
    # CAT_07: 건설재해예방전문지도기관 기술지도비 (공식 서식 8페이지)
    "CAT_07": {
        "date": ["점검일"],
        "description": ["지도항목", "사 용 내 역", "사용내역", "사용 내역"],
        "amount": ["소요 경비", "소요경비", "금액"],
        "extra": {
            "institution": ["지도기관"],
        },
    },
    # CAT_08: 본사 전담조직 근로자 임금 등 (공식 서식 9페이지)
    "CAT_08": {
        "date": ["지급일"],
        "description": ["지급내역", "지급 내역", "사 용 내 역", "사용내역", "사용 내역"],
        "amount": ["지급액", "금액"],
        "extra": {
            "org": ["소속"],
            "position": ["직책"],
            "name": ["성명"],
        },
    },
    # CAT_09: 위험성평가 등에 따른 소요비용 (공식 서식 10페이지)
    "CAT_09": {
        "date": ["사용일", "사용일자"],
        "description": ["품목명", "사 용 내 역", "사용내역", "사용 내역", "품 목", "품목"],
        "amount": ["금액", "소요 비용", "소요비용"],
        "extra": {
            "decision_date": ["결정일"],
            "unit_price": ["단가"],
            "qty": ["수량"],
        },
    },
}

# 단순 서식용 공통 헤더 키워드
SIMPLE_HEADER_MAP = {
    "date": ["사용일자", "사용일", "일자", "날짜", "지급일", "교육일", "점검일"],
    "category": ["항목", "항 목", "항목명", "구분", "분류"],
    "description": [
        "사용내역",
        "내역",
        "품목",
        "품명",
        "사용내용",
        "교육과목",
        "지도항목",
    ],
    "unit": ["단위"],
    "qty": ["수량"],
    "unit_price": ["단가"],
    "amount": ["금액", "사용금액", "소요경비", "소요 경비", "지급금액", "지급액"],
    "note": ["비고", "비 고"],
}

# 소계/합계 행 감지 키워드
SKIP_KEYWORDS = ["합계", "소계", "총계", "계상액", "전월까지", "누계", "금 월", "금월"]

# extra 필드 중 최상위 레벨로 올릴 표준 필드
STANDARD_EXTRA_FIELDS = {"unit", "qty", "unit_price"}


# ══════════════════════════════════════════════════════════
# 1. 유틸리티
# ══════════════════════════════════════════════════════════


def is_usage_statement(file_bytes: bytes) -> bool:
    """
    PDF 바이트를 읽어 사용내역서 여부를 판별한다.

    첫 페이지에서 텍스트를 빠르게 추출 후 USAGE_STATEMENT_KEYWORDS 중
    MIN_KEYWORD_MATCH개 이상 발견되면 사용내역서로 판단한다.

    파싱(parse_pdf) 전에 호출해 잘못된 파일 업로드를 조기에 차단한다.

    키워드 조정: parse_usage_statement.py 상단 USAGE_STATEMENT_KEYWORDS 수정
    """
    import io
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = pdf.pages[0].extract_text() or "" if pdf.pages else ""
        matched = sum(1 for kw in USAGE_STATEMENT_KEYWORDS if kw in text)
        return matched >= MIN_KEYWORD_MATCH
    except Exception:
        return False


def clean(v) -> str:
    """셀 값 → 공백 제거 문자열"""
    return str(v).strip().replace("\n", " ") if v is not None else ""


def parse_amount(value) -> int | None:
    """문자열 금액 → 정수 변환 (total_amount용)"""
    s = clean(value).replace(",", "").replace("원", "").replace(" ", "")
    if not s or s in ["-", "─", "—", "0.0", "0"]:
        return None
    try:
        v = int(float(s))
        return v if v > 0 else None
    except ValueError:
        return None


def parse_number(value) -> float | None:
    """문자열 숫자 → float 변환 (quantity, unit_price용)
    BUG FIX: "30개", "15롤", "20켤레" 등 단위 포함 값에서 앞 숫자만 추출
    """
    s = clean(value).replace(",", "").replace("원", "").replace(" ", "")
    if not s or s in ["-", "─", "—"]:
        return None
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        m = re.match(r"^([\d.]+)", s)
        if m:
            try:
                v = float(m.group(1))
                return v if v > 0 else None
            except ValueError:
                pass
        return None


def parse_date(value) -> str | None:
    """다양한 날짜 형식 → YYYY-MM-DD"""
    s = clean(value)
    if not s:
        return None
    patterns = [
        r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})",
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
        r"(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            if len(y) == 2:
                y = "20" + y
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    return None


def is_skip_row(row: list) -> bool:
    """소계/합계/헤더 반복 행 감지"""
    zone = " ".join(clean(c) for c in row[:4]).replace(" ", "")
    return any(kw.replace(" ", "") in zone for kw in SKIP_KEYWORDS)


def find_col(row: list, keywords: list) -> int | None:
    """키워드 목록으로 컬럼 인덱스 탐색"""
    for i, cell in enumerate(row):
        cell_clean = clean(cell).replace(" ", "")
        for kw in keywords:
            if kw.replace(" ", "") in cell_clean:
                return i
    return None


def recalc_total(unit_price: float | None, quantity: float | None, parsed_amount: int | None) -> int | None:
    """
    total_amount 결정 — 문서 기재 금액 우선, 없을 때만 단가×수량으로 보완
    BUG FIX: 단가×수량을 항상 우선하면 부가세 포함 금액을 잃어버리는 문제 방지
    """
    if parsed_amount is not None:
        return parsed_amount
    if unit_price and quantity:
        return round(unit_price * quantity)
    return None


def build_remark(extra_data: dict) -> str | None:
    """표준 필드 외 나머지 extra → remark 문자열로 병합"""
    parts = [
        f"{k}:{v}"
        for k, v in extra_data.items()
        if k not in STANDARD_EXTRA_FIELDS and v
    ]
    return " | ".join(parts) if parts else None


# ══════════════════════════════════════════════════════════
# 2. 서식 감지
# ══════════════════════════════════════════════════════════


def detect_format(pdf) -> str:
    """
    공식 서식(별지 제1호) vs 단순 서식 판별.
    반환: 'official' | 'simple'
    """
    if len(pdf.pages) >= 9:
        first_text = pdf.pages[0].extract_text() or ""
        if "별지" in first_text or "항 목 별 사 용 내 역" in (
            pdf.pages[1].extract_text() or ""
        ):
            return "official"
    return "simple"


# ══════════════════════════════════════════════════════════
# 3. 헤더(요약) 페이지 파싱
# ══════════════════════════════════════════════════════════

META_PATTERNS = {
    "건설업체명": [r"건설업체명\s*[:\s]*([^\s공소대발계]{2,30})"],
    "공사명": [r"공\s*사\s*명\s*[:\s]*([^\n]{2,50})"],
    "소재지": [r"소\s*재\s*지\s*[:\s]*([^\n]{2,50})"],
    "대표자": [r"대\s*표\s*자\s*[:\s]*([^\n]{1,20})"],
    "발주자": [r"발\s*주\s*자\s*[:\s]*([^\n]{1,30})"],
    # 공식 서식은 라벨을 "누 계 공 정 율"처럼 글자 사이를 띄우고, 표기도 률/율이 혼용된다.
    # 글자 간 공백과 률·율 변형을 모두 허용해야 70% 등의 값을 잡을 수 있다.
    "공정률": [
        r"누\s*계\s*공\s*정\s*[률율]\s*[:\s]*([\d.]+)\s*%?",
        r"공\s*정\s*[률율]\s*[:\s]*([\d.]+)\s*%?",
    ],
    "계상된안전관리비": [r"계\s*상\s*된\s*안전관리비\s*[:\s]*([\d,]+)"],
    "공사금액": [r"공\s*사\s*금\s*액\s*([\d,]+)\s*원"],
}


# 표(key-value) 셀에서 직접 읽을 헤더 텍스트 메타 (라벨 → header 필드).
# extract_text 정규식은 표 레이아웃에서 라벨/값을 흩어 놓아(예: 공사명이 칸 안에서
# 줄바꿈되어 "…빌딩"과 "신축공사"로 분리) 값을 일부만 잡는다. 표 셀은 칸 단위로
# 묶여 있어 더 정확하므로, 변별 텍스트 필드는 표 값으로 보강·교정한다.
# (금액·공정률 등 숫자 필드는 단위가 섞여 있어 기존 정규식을 그대로 사용)
_TABLE_LABEL_TO_FIELD = {
    "공사명": "공사명",
    "소재지": "소재지",
    "대표자": "대표자",
    "발주자": "발주자",
    "수급인": "건설업체명",
    "건설업체명": "건설업체명",
}


def _extract_meta_from_tables(tables) -> dict:
    """page.extract_tables() 결과에서 라벨 옆 셀을 읽어 헤더 텍스트 메타를 추출한다."""
    out: dict = {}
    for table in tables or []:
        for row in table or []:
            if not row:
                continue
            for i in range(len(row) - 1):
                label = "".join((row[i] or "").split())
                field = _TABLE_LABEL_TO_FIELD.get(label)
                if not field or field in out:
                    continue
                value = row[i + 1]
                if value is None:
                    continue
                value = " ".join(str(value).split())  # 셀 내 줄바꿈·다중 공백 정리
                if value:
                    out[field] = value
    return out


def parse_header_page(page) -> dict:
    """1페이지 텍스트 + 테이블에서 헤더 메타 추출"""
    text = page.extract_text() or ""
    header = {k: None for k in META_PATTERNS}

    for key, patterns in META_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                header[key] = m.group(1).strip()
                break

    # category_summaries용: 0원도 포함해야 하므로 parse_amount 대신 직접 파싱
    def _parse_amount_or_zero(v) -> int | None:
        s = clean(v).replace(",", "").replace("원", "").replace(" ", "")
        if not s or s in ["-", "─", "—"]:
            return None
        try:
            return max(0, int(float(s)))
        except ValueError:
            return None

    # 행 번호(1~9) → 카테고리 코드
    NUMBER_TO_CATEGORY = {str(i): f"CAT_0{i}" for i in range(1, 10)}

    summaries = []
    tables = page.extract_tables()

    # 표 셀 기반 메타 보강: 표에서 직접 읽은 값이 더 정확하므로 텍스트 필드를 덮어쓴다.
    # (특히 공사명이 칸 안에서 줄바꿈돼 정규식이 일부만 잡는 문제를 해결)
    for field, value in _extract_meta_from_tables(tables).items():
        header[field] = value

    for table in tables:
        if not table or not table[0]:
            continue

        # BUG FIX: "항 목" 헤더를 가진 요약 테이블만 처리 (상위 2행에서 확인)
        header_text = "".join(
            clean(c) for row in table[:2] if row for c in row if c
        ).replace(" ", "")
        if "항목" not in header_text:
            continue

        for row in table[1:]:
            if not row or is_skip_row(row):
                continue

            # 첫 번째 셀의 번호(예: "1.", "2.") → 카테고리 코드
            first_cell = clean(row[0]) if row[0] else ""
            m = re.match(r"^(\d+)[.\s]", first_cell.strip())
            if not m:
                continue

            cat_code = NUMBER_TO_CATEGORY.get(m.group(1))
            if not cat_code:
                continue

            # BUG FIX: 0원 항목도 포함 (CAT_08, CAT_09 등)
            amounts = [_parse_amount_or_zero(c) for c in row]
            amounts = [a for a in amounts if a is not None]
            if len(amounts) >= 2:
                summaries.append(
                    {
                        "항목코드": cat_code,
                        "항목명": CATEGORY_NAME_MAP.get(cat_code, ""),
                        "전회금액": amounts[-3] if len(amounts) >= 3 else 0,
                        "금회금액": amounts[-2],
                        "누계금액": amounts[-1],
                    }
                )

    header["category_summaries"] = summaries
    return header


# ══════════════════════════════════════════════════════════
# 4. 공식 서식 - 카테고리 페이지 파싱
# ══════════════════════════════════════════════════════════


def parse_category_page(page, category_code: str, page_no: int) -> list:
    """카테고리 코드에 맞는 컬럼 정의로 테이블 파싱"""
    col_def = CATEGORY_COLUMNS.get(category_code, {})
    date_kws = col_def.get("date", [])
    desc_kws = col_def.get("description", [])
    amount_kws = col_def.get("amount", [])
    extra_defs = col_def.get("extra", {})

    items = []
    line_no = 0
    tables = page.extract_tables()

    for table in tables:
        if not table:
            continue

        # 헤더 행 탐색 (상위 4행)
        header_row = None
        header_idx = -1
        for i, row in enumerate(table[:4]):
            if (
                find_col(row, date_kws) is not None
                or find_col(row, amount_kws) is not None
            ):
                header_row = row
                header_idx = i
                break

        if header_row is None:
            continue

        # 컬럼 인덱스 확정
        date_col = find_col(header_row, date_kws)
        desc_col = find_col(header_row, desc_kws)
        amount_col = find_col(header_row, amount_kws)
        extra_cols = {
            field: find_col(header_row, kws) for field, kws in extra_defs.items()
        }

        # 데이터 행 파싱
        for row in table[header_idx + 1 :]:
            if not row or is_skip_row(row):
                continue

            row_c = [clean(c) for c in row]
            amount = parse_amount(row_c[amount_col]) if amount_col is not None else None
            if amount is None:
                continue

            line_no += 1

            # extra 필드 수집
            extra_data = {}
            for field, col_idx in extra_cols.items():
                if col_idx is not None and col_idx < len(row_c) and row_c[col_idx]:
                    extra_data[field] = row_c[col_idx]

            unit_price = parse_number(extra_data.get("unit_price"))
            quantity   = parse_number(extra_data.get("qty"))

            item = {
                "line_id": str(uuid.uuid4()),
                "category_code": category_code,
                "used_on": parse_date(row_c[date_col])
                if date_col is not None
                else None,
                "item_name": row_c[desc_col]
                if desc_col is not None and row_c[desc_col]
                else None,
                "unit": extra_data.get("unit"),
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": recalc_total(unit_price, quantity, amount),
                "remark": build_remark(extra_data),
                "page_no": page_no,
                "line_no": line_no,
            }

            items.append(item)

    return items


# ══════════════════════════════════════════════════════════
# 5. 단순 서식 파싱
# ══════════════════════════════════════════════════════════


def parse_simple_format(pdf) -> tuple[dict, list]:
    """단순 서식 전체 파싱 → (헤더, 라인아이템 목록)"""
    all_text = ""
    items = []

    # BUG FIX: page 1은 parse_header_page()로 처리해 category_summaries 추출
    header_data = parse_header_page(pdf.pages[0])
    all_text += (pdf.pages[0].extract_text() or "") + "\n"

    # BUG FIX: page 2부터 line_items 파싱 (page 1 요약 테이블 line_items 유입 방지)
    for page_num, page in enumerate(pdf.pages[1:], 2):
        page_text = page.extract_text() or ""
        all_text += page_text + "\n"

        line_no = 0
        tables = page.extract_tables()

        for table in tables:
            if not table:
                continue

            # BUG FIX: 요약 테이블 감지 → 스킵 ("전월까지사용금액" 등 헤더 존재 시)
            top2_text = " ".join(
                clean(c) for row in table[:2] if row for c in row if c
            ).replace(" ", "")
            if any(kw in top2_text for kw in ["전월까지", "금월사용", "누계사용", "전월까지사용"]):
                continue

            # 헤더 행 탐색
            header_row = None
            header_idx = -1
            for i, row in enumerate(table[:5]):
                mapping = {
                    k: find_col(row, kws) for k, kws in SIMPLE_HEADER_MAP.items()
                }
                if sum(1 for v in mapping.values() if v is not None) >= 2:
                    header_row = row
                    header_idx = i
                    col_map = mapping
                    break

            if header_row is None:
                continue

            # BUG FIX: 카테고리 carry-forward (항목 첫 행에만 카테고리명 있는 경우 대응)
            last_cat_code = None

            for row in table[header_idx + 1 :]:
                if not row or is_skip_row(row):
                    continue

                row_c = [clean(c) for c in row]
                amount = (
                    parse_amount(row_c[col_map["amount"]])
                    if col_map.get("amount") is not None
                    else None
                )
                if amount is None:
                    continue

                line_no += 1

                category_raw = (
                    row_c[col_map["category"]]
                    if col_map.get("category") is not None
                    else None
                )
                if category_raw:
                    inferred = _infer_category_code(category_raw)
                    if inferred:
                        last_cat_code = inferred
                    cat_code = inferred or last_cat_code
                else:
                    cat_code = last_cat_code

                unit_price = parse_number(row_c[col_map["unit_price"]]) if col_map.get("unit_price") is not None else None
                quantity   = parse_number(row_c[col_map["qty"]]) if col_map.get("qty") is not None else None

                items.append(
                    {
                        "line_id": str(uuid.uuid4()),
                        "category_code": cat_code,
                        "used_on": parse_date(row_c[col_map["date"]])
                        if col_map.get("date") is not None
                        else None,
                        "item_name": row_c[col_map["description"]]
                        if col_map.get("description") is not None
                        else None,
                        "unit": row_c[col_map["unit"]]
                        if col_map.get("unit") is not None
                        else None,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total_amount": recalc_total(unit_price, quantity, amount),
                        "remark": row_c[col_map["note"]]
                        if col_map.get("note") is not None
                        else None,
                        "page_no": page_num,
                        "line_no": line_no,
                    }
                )

    # 헤더 메타 보완 (텍스트 기반 추출이 더 정확한 필드만 덮어씀)
    text_header = _extract_meta_from_text(all_text)
    for k, v in text_header.items():
        if k == "category_summaries":
            continue
        if header_data.get(k) is None and v is not None:
            header_data[k] = v

    return header_data, items


def _infer_category_code(text: str) -> str | None:
    """카테고리 텍스트에서 코드 추론"""
    # BUG FIX 1: 숫자 번호(예: "1.", "7.") 기반 직접 매핑 — 텍스트 키워드보다 먼저 시도
    NUMBER_TO_CAT = {str(i): f"CAT_0{i}" for i in range(1, 10)}
    m = re.match(r"^(\d)[.\s]", text.strip())
    if m:
        code = NUMBER_TO_CAT.get(m.group(1))
        if code:
            return code

    # BUG FIX 2: 공백 제거 후 키워드 비교 (예: "기술 지도비" vs "기술지도비")
    text_nospace = text.replace(" ", "")
    for code, name in CATEGORY_NAME_MAP.items():
        keywords = name.replace("등", "").replace("·", "").replace(" ", "").split()
        if any(kw in text_nospace for kw in keywords if len(kw) > 1):
            return code
    return None


def _extract_meta_from_text(text: str) -> dict:
    """전체 텍스트에서 메타 정보 추출"""
    header = {k: None for k in META_PATTERNS}
    for key, patterns in META_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                header[key] = m.group(1).strip()
                break
    header["category_summaries"] = []
    return header


# ══════════════════════════════════════════════════════════
# 6. 메인 파싱
# ══════════════════════════════════════════════════════════


def parse_pdf(pdf_path: str) -> dict:
    """사용내역서 PDF 전체 파싱 → 표준 결과 딕셔너리"""
    result = {
        "document_id": str(uuid.uuid4()),
        "source_file": Path(pdf_path).name,
        "parsed_at": datetime.now().isoformat(),
        "format": None,
        "header": {},
        "category_summaries": [],
        "line_items": [],
        "parse_status": "FAILED",
        "warnings": [],
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            fmt = detect_format(pdf)
            result["format"] = fmt

            if fmt == "official":
                header_data = parse_header_page(pdf.pages[0])
                result["header"] = {
                    k: v for k, v in header_data.items() if k != "category_summaries"
                }
                result["category_summaries"] = header_data.get("category_summaries", [])

                for page_num, page in enumerate(pdf.pages[1:], 2):
                    cat_code = PAGE_CATEGORY_MAP.get(page_num)
                    if cat_code is None:
                        # 요약 페이지(1) 외에 스킵 사유가 있는 페이지는 경고 추가
                        skip_reason = PAGE_SKIP_REASONS.get(page_num)
                        if skip_reason:
                            result["warnings"].append(
                                f"페이지 {page_num} 스킵: {skip_reason}"
                            )
                        continue
                    items = parse_category_page(page, cat_code, page_num)
                    result["line_items"].extend(items)
                    if not items:
                        result["warnings"].append(
                            f"페이지 {page_num} ({CATEGORY_NAME_MAP.get(cat_code)}): 파싱된 항목 없음"
                        )

            else:
                header, items = parse_simple_format(pdf)
                result["category_summaries"] = header.pop("category_summaries", [])
                result["header"] = header
                result["line_items"] = items

            result["parse_status"] = "SUCCESS" if result["line_items"] else "PARTIAL"

    except Exception as e:
        result["parse_status"] = "FAILED"
        result["warnings"].append(f"파싱 오류: {str(e)}")

    return result


# ══════════════════════════════════════════════════════════
# 7. 저장
# ══════════════════════════════════════════════════════════


def save_json(parsed: dict, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
# 9. 콘솔 요약
# ══════════════════════════════════════════════════════════


def print_summary(parsed: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  파일:       {parsed.get('source_file')}")
    print(f"  서식:       {parsed.get('format')}")
    print(f"  파싱 상태:  {parsed.get('parse_status')}")
    print(f"{sep}")

    items = parsed.get("line_items", [])
    if items:
        total = sum(i.get("total_amount", 0) for i in items)
        print(f"  총 항목수:  {len(items)}건")
        print(f"  총 금액:    {total:,}원")

        from collections import Counter

        cats = Counter(i.get("category_code") for i in items)
        print("\n  [카테고리별 항목 수]")
        for code, cnt in sorted(cats.items(), key=lambda x: x[0] or ""):
            name = CATEGORY_NAME_MAP.get(code, f"항목 {code}")
            print(f"     {code}. {name[:25]:<25}  {cnt}건")

    warnings = parsed.get("warnings", [])
    if warnings:
        print("\n  [경고]")
        for w in warnings:
            print(f"  ⚠️  {w}")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════
# 10. 메인
# ══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="산업안전보건관리비 사용내역서 PDF 파싱 (JSON 출력)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="사용내역서 PDF 파일 경로")
    group.add_argument("--folder", help="PDF 폴더 경로 (배치 처리)")
    parser.add_argument("--output", default=None, help="결과 저장 폴더")
    args = parser.parse_args()

    def _process(pdf_path: str, out_dir: str):
        print(f"\n처리 중: {Path(pdf_path).name}")
        parsed = parse_pdf(pdf_path)
        print_summary(parsed)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(pdf_path).stem

        json_path = Path(out_dir) / f"{stem}_{ts}.json"
        save_json(parsed, str(json_path))
        print(f"  JSON 저장: {json_path}")

        return parsed

    if args.pdf:
        if not Path(args.pdf).exists():
            print(f"[오류] 파일 없음: {args.pdf}")
            return
        out_dir = args.output or str(Path(args.pdf).parent / "parsed_results")
        _process(args.pdf, out_dir)

    elif args.folder:
        folder = Path(args.folder)
        pdfs = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
        if not pdfs:
            print(f"[오류] PDF 없음: {folder}")
            return
        out_dir = args.output or str(folder / "parsed_results")
        print(f"\n배치 처리: {len(pdfs)}개 PDF")
        for pdf in sorted(pdfs):
            _process(str(pdf), out_dir)


if __name__ == "__main__":
    main()
