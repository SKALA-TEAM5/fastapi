# --------------------------------------------------------------------------
# 작성자   : 이현수(kacalu0930)
# 작성일   : 2026-05-11
# 수정일   : 2026-06-18 (Clova→VLM 전환 + 세금계산서 PDF·이미지 모두 VLM 처리로 기능변경)
#
# [ 주요 함수 정의 ]  ※ 실제 호출되는 함수만 기재
#
# 1. parse_tax_invoice() : 세금계산서 파싱 메인 진입 — 확장자 무관 VLM 처리
# 2. parse_with_vlm()    : VLM(call_vision) 호출 + 표준 스키마 매핑 (PDF·이미지 공통)
# 3. parse_from_pdf()    : PDF pdfplumber 텍스트 추출 파싱 (거래명세표/임금명세서 PDF에서 사용)
# 4. _extract_fields_from_text() : 추출 텍스트 → 공급자/공급받는자/품목/금액 필드 파싱
# 5. _validate()         : 세금계산서 결과 검증(필수 필드 / 금액 정합)
# 6. (CLI) main() / process_folder() / process_single() / save_result() / print_summary()
#
# [ 처리 단계 ]
#   parse_tax_invoice → (이미지·PDF) parse_with_vlm → call_vision → 표준 스키마 매핑 → _validate
# --------------------------------------------------------------------------
"""
산업안전관리비 세금계산서 파싱 모듈 v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파일 확장자를 자동 감지해 처리 방식을 분기한다.
  - .pdf              → pdfplumber 텍스트 직접 추출 (parse_usage_statement.py 동일 방식)
  - .jpg/.jpeg/.png/.tif/.tiff → VLM(Gemini/OpenAI) 구조화 추출 (vlm_ocr.call_vision)

두 경로 모두 처리 후 동일한 JSON 구조로 출력.

사용법:
    # 단일 파일 (PDF 또는 이미지)
    python parse_tax_invoice.py --file 세금계산서.pdf
    python parse_tax_invoice.py --file 세금계산서.jpg

    # 폴더 배치 처리
    python parse_tax_invoice.py --folder invoices/

    # 출력 폴더 지정
    python parse_tax_invoice.py --file 세금계산서.pdf --output ./results/

설치:
    pip install pdfplumber python-dotenv

환경변수 (이미지 처리 = VLM, .env 파일 또는 시스템 환경변수):
    VLM_PROVIDER=gemini | openai
    GEMINI_API_KEY / OPENAI_API_KEY (선택한 프로바이더 키)

    .env.example 파일을 복사하여 .env 파일을 생성하고 키를 입력하세요.
"""

import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수 직접 설정 필요

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# ══════════════════════════════════════════════
# 0. 설정 — 지원 확장자
# ══════════════════════════════════════════════

# [Clova→VLM 리팩토링] 여기 있던 CLOVA_SECRET / CLOVA_URL 환경변수 상수 삭제,
#   관련 import(os, uuid, requests)도 제거.

# 지원 확장자
PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALL_EXTS   = PDF_EXTS | IMAGE_EXTS


# ══════════════════════════════════════════════
# 1. 공통 유틸리티
# ══════════════════════════════════════════════

def _safe_int(value) -> int | None:
    """문자열 → 정수 변환 (쉼표, 공백, 원 제거)"""
    if value is None:
        return None
    try:
        cleaned = str(value).replace(",", "").replace(" ", "").replace("원", "").strip()
        return int(float(cleaned)) if cleaned else None
    except (ValueError, TypeError):
        return None


def _normalize_biz_num(text: str) -> str | None:
    """
    사업자등록번호 정규화 → 'NNN-NN-NNNNN' 형식.
    입력 예: '1234567890', '123-45-67890', '123 45 67890'
    """
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return text.strip() or None


def _parse_date(text: str) -> str | None:
    """다양한 날짜 형식 → 'YYYY-MM-DD'"""
    if not text:
        return None
    patterns = [
        r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})",   # 2024.03.15 / 2024-03-15
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",    # 2024년 3월 15일
        r"(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})",    # 24.03.15
    ]
    for pat in patterns:
        m = re.search(pat, str(text))
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            if len(y) == 2:
                y = "20" + y
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    return None


def _empty_result(file_path: str, file_type: str, parse_method: str) -> dict:
    """기본 빈 결과 구조 반환"""
    return {
        "file_path":          str(file_path),
        "file_type":          file_type,
        "parse_method":       parse_method,
        "supplier": {
            "company_name":   None,
            "business_number": None,
            "representative": None,
        },
        "buyer": {
            "company_name":   None,
            "business_number": None,
        },
        "issue_date":         None,
        "items":              [],
        "total_supply_amount": None,
        "total_tax_amount":   None,
        "total_amount":       None,
        "raw_text":           "",
        "validation": {
            "is_valid": False,
            "warnings": [],
        },
    }


# ══════════════════════════════════════════════
# 2. 텍스트 기반 세금계산서 필드 추출
#    (PDF 텍스트 추출 결과 및 OCR raw_text 공통 사용)
# ══════════════════════════════════════════════

# 공급자 / 공급받는자 키워드
_SUPPLIER_KEYWORDS = ["공급자", "공급 자", "판매자", "발행자"]
_BUYER_KEYWORDS    = ["공급받는자", "공급 받는 자", "구매자", "매입자"]

# 발행일 키워드
_DATE_KEYWORDS     = ["작성일자", "발행일자", "발행일", "작성일", "공급일자", "거래일자"]


def _extract_fields_from_text(text: str) -> dict:
    """
    세금계산서 텍스트에서 주요 필드를 정규식으로 추출.

    Returns:
        {
          supplier, buyer, issue_date, items,
          total_supply_amount, total_tax_amount, total_amount
        }
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)

    # ── 발행일 추출 ─────────────────────────────────────────
    issue_date = None
    for kw in _DATE_KEYWORDS:
        pat = rf"{re.escape(kw)}\s*[：:\s]\s*(\d{{4}}[.\-/년]\s*\d{{1,2}}[.\-/월]\s*\d{{1,2}}일?)"
        m = re.search(pat, joined)
        if m:
            issue_date = _parse_date(m.group(1))
            if issue_date:
                break

    # 날짜를 못 찾았으면 전문에서 'YYYY.MM.DD' 패턴 첫 번째 매칭
    if not issue_date:
        m = re.search(r"(\d{4})[.\-/](\d{2})[.\-/](\d{2})", joined)
        if m:
            issue_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # ── 사업자등록번호 추출 ──────────────────────────────────
    biz_nums = re.findall(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{5}\b", joined)
    biz_nums = [_normalize_biz_num(b) for b in biz_nums]

    # ── 상호(공급자/공급받는자) 추출 ────────────────────────
    supplier_name = None
    buyer_name    = None

    # 패턴: '공급자 상호 OOO주식회사' 또는 '공급자\n상호 OOO' 등
    for kw in _SUPPLIER_KEYWORDS:
        pat = rf"{re.escape(kw)}.{{0,30}}상호\s*[：:\s]\s*([^\s,;|]+(?:\s[^\s,;|]+){{0,3}})"
        m = re.search(pat, joined)
        if m:
            supplier_name = m.group(1).strip()
            break
    if not supplier_name:
        for kw in _SUPPLIER_KEYWORDS:
            # 줄 단위에서 키워드 근처 상호 탐색
            for i, line in enumerate(lines):
                if kw in line and i + 1 < len(lines):
                    cand = lines[i + 1].strip()
                    if cand and not re.match(r"^\d", cand):
                        supplier_name = cand[:30]
                        break
            if supplier_name:
                break

    for kw in _BUYER_KEYWORDS:
        pat = rf"{re.escape(kw)}.{{0,30}}상호\s*[：:\s]\s*([^\s,;|]+(?:\s[^\s,;|]+){{0,3}})"
        m = re.search(pat, joined)
        if m:
            buyer_name = m.group(1).strip()
            break

    # ── 대표자 추출 ─────────────────────────────────────────
    representative = None
    m = re.search(r"대표자?\s*[：:\s]\s*([가-힣]{2,5}(?:\s[가-힣]{1,5})?)", joined)
    if m:
        representative = m.group(1).strip()

    # ── 공급자 사업자번호 / 공급받는자 사업자번호 ─────────────
    supplier_biz = biz_nums[0] if len(biz_nums) >= 1 else None
    buyer_biz    = biz_nums[1] if len(biz_nums) >= 2 else None

    # ── 합계금액 추출 ────────────────────────────────────────
    total_supply = None
    total_tax    = None
    total_amount = None

    # 공급가액 / 세액 / 합계금액 패턴
    _amount_kws = [
        ("total_supply", ["공급가액합계", "공급가액 합계", "공급가액"]),
        ("total_tax",    ["세액합계", "세액 합계", "세액"]),
        ("total_amount", ["합계금액", "합 계", "합계", "청구금액", "총액"]),
    ]
    for field, kws in _amount_kws:
        for kw in kws:
            pat = rf"{re.escape(kw)}\s*[：:\s]?\s*([\d,]+)"
            m = re.search(pat, joined)
            if m:
                val = _safe_int(m.group(1))
                if val:
                    if field == "total_supply":
                        total_supply = val
                    elif field == "total_tax":
                        total_tax = val
                    elif field == "total_amount":
                        total_amount = val
                    break

    # 합계금액이 없고 공급가액+세액가 있으면 합산
    if total_amount is None and total_supply and total_tax:
        total_amount = total_supply + total_tax

    # ── 품목 행 파싱 ─────────────────────────────────────────
    items = _parse_items_from_lines(lines)

    return {
        "supplier": {
            "company_name":    supplier_name,
            "business_number": supplier_biz,
            "representative":  representative,
        },
        "buyer": {
            "company_name":    buyer_name,
            "business_number": buyer_biz,
        },
        "issue_date":          issue_date,
        "items":               items,
        "total_supply_amount": total_supply,
        "total_tax_amount":    total_tax,
        "total_amount":        total_amount,
    }


def _parse_items_from_lines(lines: list) -> list:
    """
    텍스트 줄에서 세금계산서 품목 행 파싱.

    세금계산서 품목 행의 전형적 패턴:
      품목명  규격  수량  단가  공급가액  세액
    금액 2~3개가 같은 줄에 있는 행을 품목으로 간주.
    """
    items = []
    # 헤더 행으로 보이는 줄 인덱스 기록 (건너뜀)
    header_pats = re.compile(
        r"품목|품명|규격|수량|단가|공급가액|세액|월|일|비고", re.IGNORECASE
    )
    skip_pats = re.compile(
        r"합계|소계|총계|^$", re.IGNORECASE
    )

    for line in lines:
        # 헤더/합계 줄 제외
        if skip_pats.search(line):
            continue

        # 금액 숫자 2개 이상 있는 줄만 품목 행으로 간주
        amounts = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b", line)
        int_amounts = [_safe_int(a) for a in amounts if _safe_int(a) and _safe_int(a) >= 100]
        if len(int_amounts) < 2:
            continue

        # 품목명: 줄에서 숫자·특수문자 제외한 앞쪽 한글/영문 부분
        name_m = re.match(r"^([가-힣A-Za-z()·\s\-\.]+)", line.strip())
        item_name = name_m.group(1).strip() if name_m else None

        # 품목명 없거나 너무 짧으면 건너뜀
        if not item_name or len(item_name) < 2:
            continue

        # 숫자 목록에서 수량(작은 정수) / 단가 / 공급가액 / 세액 추론
        # 정렬: 작은 수부터
        sorted_amts = sorted(int_amounts)
        quantity    = sorted_amts[0] if sorted_amts[0] < 10000 else None
        supply_amt  = int_amounts[-2] if len(int_amounts) >= 2 else None
        tax_amt     = int_amounts[-1] if len(int_amounts) >= 1 else None
        unit_price  = None

        # 단가 추론: 수량이 있고 공급가액이 있으면 역산
        if quantity and supply_amt and quantity > 0:
            unit_price = supply_amt // quantity

        items.append({
            "item_name":     item_name,
            "quantity":      quantity,
            "unit_price":    unit_price,
            "supply_amount": supply_amt,
            "tax_amount":    tax_amt,
        })

    return items


# ══════════════════════════════════════════════
# 3. PDF 경로 — pdfplumber 텍스트 추출
# ══════════════════════════════════════════════

def parse_from_pdf(file_path: str) -> dict:
    """
    PDF 세금계산서 파싱 (pdfplumber 사용).

    Args:
        file_path: PDF 파일 경로

    Returns:
        표준 세금계산서 JSON dict
    """
    result = _empty_result(file_path, "pdf", "text_extract")

    if pdfplumber is None:
        result["validation"]["warnings"].append(
            "pdfplumber 미설치 — pip install pdfplumber 실행 필요"
        )
        return result

    try:
        all_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"

        result["raw_text"] = all_text.strip()

        # 필드 추출
        extracted = _extract_fields_from_text(all_text)
        result["supplier"]           = extracted["supplier"]
        result["buyer"]              = extracted["buyer"]
        result["issue_date"]         = extracted["issue_date"]
        result["items"]              = extracted["items"]
        result["total_supply_amount"] = extracted["total_supply_amount"]
        result["total_tax_amount"]   = extracted["total_tax_amount"]
        result["total_amount"]       = extracted["total_amount"]

    except Exception as e:
        result["validation"]["warnings"].append(f"PDF 파싱 오류: {str(e)}")

    # 검증
    result = _validate(result)
    return result


# ══════════════════════════════════════════════
# 4. VLM(Gemini/OpenAI) 호출 — PDF·이미지 공통
# ══════════════════════════════════════════════
# [Clova→VLM 리팩토링] 구버전 parse_from_image(CLOVA OCR + 정규식)을 이 함수로 대체.
#   _call_clova_ocr / _extract_raw_text_from_clova / CLOVA_SECRET·URL 상수는 삭제됨.
#   call_vision 출력을 기존 표준 스키마로 매핑해 downstream 계약은 유지.

def parse_with_vlm(file_path: str) -> dict:
    """
    세금계산서 파싱 (VLM 사용, PDF·이미지 공통).

    기존 NAVER CLOVA OCR + 정규식 파싱을 대체한다.
    VLM(call_vision)에 type_hint="tax_invoice"로 파일을 전달해 구조화 JSON을 받고,
    표준 세금계산서 JSON 스키마로 매핑한다.
    → downstream(라우터·파이프라인·TaxInvoiceOCRResponse 변환) 스키마는 변하지 않는다.

    ※ PDF 입력은 VLM_PROVIDER=gemini 에서만 동작(OpenAI 경로는 PDF 미지원).

    Args:
        file_path: 세금계산서 파일 경로 (.pdf / 이미지)

    Returns:
        표준 세금계산서 JSON dict
    """
    # 지연 import: vlm_ocr ↔ parse_tax_invoice 순환 의존 방지
    from src.ocr.vlm_ocr import call_vision

    file_type = "pdf" if Path(file_path).suffix.lower() in PDF_EXTS else "image"
    result = _empty_result(file_path, file_type, "vlm")

    vlm_raw = call_vision(str(file_path), type_hint="tax_invoice")

    if "error" in vlm_raw:
        result["validation"]["warnings"].append(f"VLM 오류: {vlm_raw['error']}")
        return _validate(result)

    infer_result = vlm_raw.get("infer_result", "FAILED")
    if infer_result == "FAILED":
        reason = vlm_raw.get("fail_reason") or "VLM 판독 실패"
        result["validation"]["warnings"].append(f"VLM 판독 실패: {reason}")

    # ── 공급자 / 공급받는자 ───────────────────────────────
    supplier = vlm_raw.get("supplier") or {}
    buyer    = vlm_raw.get("buyer") or {}
    result["supplier"] = {
        "company_name":    supplier.get("name"),
        "business_number": _normalize_biz_num(supplier.get("biz_num") or ""),
        "representative":  None,  # VLM 미추출 필드
    }
    result["buyer"] = {
        "company_name":    buyer.get("name"),
        "business_number": _normalize_biz_num(buyer.get("biz_num") or ""),
    }

    # ── 작성일자 (VLM은 YYYY-MM-DD로 반환, _parse_date로 재정규화) ──
    result["issue_date"] = _parse_date(vlm_raw.get("date") or "")

    # ── 품목 (VLM 키 → 표준 키 매핑) ──────────────────────
    items = []
    for it in (vlm_raw.get("items") or []):
        items.append({
            "item_name":     it.get("name"),
            "quantity":      _safe_int(it.get("count")),
            "unit_price":    _safe_int(it.get("unit_price")),
            "supply_amount": _safe_int(it.get("amount")),
            "tax_amount":    _safe_int(it.get("tax_amount")),
        })
    result["items"] = items

    # ── 합계 금액 ─────────────────────────────────────────
    total_amount = _safe_int(vlm_raw.get("total_amount"))
    total_tax    = _safe_int(vlm_raw.get("tax_amount"))
    result["total_amount"]     = total_amount
    result["total_tax_amount"] = total_tax

    # 공급가액 합계: VLM 미제공 → 합계금액 − 세액, 둘 다 없으면 품목 공급가액 합산
    if total_amount is not None and total_tax is not None:
        result["total_supply_amount"] = total_amount - total_tax
    else:
        items_supply = sum(i["supply_amount"] or 0 for i in items)
        result["total_supply_amount"] = items_supply or None

    return _validate(result)


# ══════════════════════════════════════════════
# 5. 검증 로직
# ══════════════════════════════════════════════

def _validate(result: dict) -> dict:
    """
    세금계산서 파싱 결과 후처리 검증.
    - 필수 필드 누락 확인
    - 공급가액 + 세액 = 합계금액 검증
    - 품목 합산 vs 공급가액합계 검증
    """
    warnings = list(result["validation"].get("warnings", []))

    # ── 필수 필드 확인 ───────────────────────────────────────
    missing = []
    required = {
        "supplier.business_number": result["supplier"].get("business_number"),
        "issue_date":               result.get("issue_date"),
        "total_amount":             result.get("total_amount"),
    }
    for field, val in required.items():
        if not val:
            missing.append(field)

    if missing:
        warnings.append(f"필수 필드 누락: {', '.join(missing)}")

    # ── 공급가액 + 세액 = 합계금액 검증 ────────────────────
    supply = result.get("total_supply_amount")
    tax    = result.get("total_tax_amount")
    total  = result.get("total_amount")
    if supply and tax and total:
        calc_total = supply + tax
        diff = abs(calc_total - total)
        if diff > 100:
            warnings.append(
                f"금액 불일치: 공급가액({supply:,}) + 세액({tax:,}) = {calc_total:,} ≠ 합계({total:,}), 차이 {diff:,}"
            )

    # ── 품목 합산 vs 공급가액합계 ───────────────────────────
    items = result.get("items", [])
    if items and supply:
        items_supply_sum = sum(
            i.get("supply_amount") or 0 for i in items
        )
        if items_supply_sum > 0 and abs(items_supply_sum - supply) > 100:
            warnings.append(
                f"품목 공급가액 합산({items_supply_sum:,}) ≠ 공급가액합계({supply:,})"
            )

    # ── 유효성 판단 ─────────────────────────────────────────
    is_valid = (len(missing) == 0)

    result["validation"] = {
        "is_valid": is_valid,
        "warnings": warnings,
    }
    return result


# ══════════════════════════════════════════════
# 6. 파일 분기 처리 (메인 API)
# ══════════════════════════════════════════════

def parse_tax_invoice(
    file_path: str,
    secret: str = None,
    url: str    = None,
) -> dict:
    """
    세금계산서 파일을 파싱해 표준 JSON을 반환하는 메인 함수.

    PDF·이미지 구분 없이 모두 VLM(Gemini/OpenAI)으로 처리한다.
      ※ PDF 입력은 VLM_PROVIDER=gemini 에서만 동작한다.
        (OpenAI 경로는 PDF 직접 입력을 지원하지 않음)

    Args:
        file_path: 세금계산서 파일 경로 (.pdf / .jpg / .jpeg / .png / .tif / .tiff)
        secret:    (deprecated) 과거 CLOVA OCR Secret. 더 이상 사용하지 않음(하위호환 유지).
        url:       (deprecated) 과거 CLOVA OCR URL. 더 이상 사용하지 않음(하위호환 유지).

    Returns:
        표준 JSON dict
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    # [기능변경] 세금계산서는 PDF·이미지 구분 없이 모두 VLM으로 처리.
    #   (이전: PDF→pdfplumber, 이미지→VLM)  ※ PDF는 VLM_PROVIDER=gemini 필요
    if ext in ALL_EXTS:
        return parse_with_vlm(str(path))

    else:
        result = _empty_result(file_path, "unknown", "none")
        result["validation"]["warnings"].append(
            f"지원하지 않는 파일 형식: {ext}  (지원: {', '.join(sorted(ALL_EXTS))})"
        )
        return result


# ══════════════════════════════════════════════
# 7. 저장 유틸리티
# ══════════════════════════════════════════════

def save_result(parsed: dict, output_path: str) -> str:
    """파싱 결과를 JSON 파일로 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return str(output_path)


# ══════════════════════════════════════════════
# 8. 콘솔 요약 출력
# ══════════════════════════════════════════════

def print_summary(parsed: dict):
    """파싱 결과 요약 터미널 출력"""
    sep = "─" * 56
    p   = Path(parsed.get("file_path", "N/A"))

    print(f"\n{sep}")
    print(f"  📄 파일:       {p.name}")
    print(f"  🔍 처리 방식:  {parsed.get('parse_method', '-')}")
    print(f"  🗂️  파일 유형:  {parsed.get('file_type', '-')}")
    print(f"{sep}")

    sup = parsed.get("supplier", {})
    buy = parsed.get("buyer", {})

    print(f"  [공급자]")
    print(f"     · 상호:       {sup.get('company_name') or '─'}")
    print(f"     · 사업자번호: {sup.get('business_number') or '─'}")
    print(f"     · 대표자:     {sup.get('representative') or '─'}")
    print(f"  [공급받는자]")
    print(f"     · 상호:       {buy.get('company_name') or '─'}")
    print(f"     · 사업자번호: {buy.get('business_number') or '─'}")
    print(f"{sep}")
    print(f"  📅 작성일자:   {parsed.get('issue_date') or '─'}")
    print(f"{sep}")

    items = parsed.get("items", [])
    if items:
        print(f"  🛒 품목 ({len(items)}개):")
        for item in items[:5]:
            name  = item.get("item_name") or "─"
            qty   = item.get("quantity")
            sup_a = item.get("supply_amount")
            tax_a = item.get("tax_amount")
            qty_str  = f"×{qty}"      if qty   else ""
            sup_str  = f"{sup_a:,}원" if sup_a else "─"
            tax_str  = f"{tax_a:,}원" if tax_a else "─"
            print(f"     · {name} {qty_str}  공급가액: {sup_str}  세액: {tax_str}")
        if len(items) > 5:
            print(f"     ... (총 {len(items)}개)")
    else:
        print(f"  🛒 품목:       인식된 항목 없음")

    print(f"{sep}")
    supply = parsed.get("total_supply_amount")
    tax    = parsed.get("total_tax_amount")
    total  = parsed.get("total_amount")
    print(f"  💰 공급가액:   {f'{supply:,}원' if supply else '─'}")
    print(f"  🧾 세액:       {f'{tax:,}원'    if tax    else '─'}")
    print(f"  💵 합계금액:   {f'{total:,}원'  if total  else '─'}")
    print(f"{sep}")

    val = parsed.get("validation", {})
    print(f"  ✅ 유효성:     {'유효' if val.get('is_valid') else '⚠️ 검토 필요'}")
    for w in val.get("warnings", []):
        print(f"  ⚠️  {w}")

    print(f"{sep}\n")


# ══════════════════════════════════════════════
# 9. 단일 / 폴더 처리
# ══════════════════════════════════════════════

def process_single(
    file_path: str,
    output_dir: str = None,
    secret: str    = None,
    url: str       = None,
) -> dict:
    """단일 세금계산서 파일 처리 → 파싱 → 저장"""
    path = Path(file_path)
    print(f"\n처리 중: {path.name}")

    parsed = parse_tax_invoice(str(path), secret=secret, url=url)

    out_dir  = Path(output_dir) if output_dir else path.parent / "tax_invoice_results"
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{path.stem}_tax_invoice_{ts}.json"
    saved    = save_result(parsed, str(out_path))

    print_summary(parsed)
    print(f"  💾 저장: {saved}")

    return parsed


def process_folder(
    folder_path: str,
    output_dir: str = None,
    secret: str    = None,
    url: str       = None,
) -> list[dict]:
    """폴더 내 세금계산서 파일 전체 배치 처리"""
    folder = Path(folder_path)
    files  = sorted([f for f in folder.iterdir() if f.suffix.lower() in ALL_EXTS])

    if not files:
        print(f"[경고] 지원 파일 없음: {folder}  (지원 확장자: {', '.join(sorted(ALL_EXTS))})")
        return []

    out_dir = Path(output_dir) if output_dir else folder / "tax_invoice_results"
    print(f"\n📂 배치 처리: {folder.name}  ({len(files)}개 파일)")
    print(f"📁 결과 저장: {out_dir}\n")

    results = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        result = process_single(str(f), str(out_dir), secret=secret, url=url)
        results.append(result)
        time.sleep(0.2)   # 연속 API 호출 간격

    # ── 배치 요약 ────────────────────────────────────────────
    valid   = sum(1 for r in results if r.get("validation", {}).get("is_valid"))
    invalid = len(results) - valid
    print(f"\n{'═'*56}")
    print(f"  배치 처리 완료: 총 {len(results)}개 | ✅ 유효 {valid}개 | ⚠️ 검토 필요 {invalid}개")
    print(f"{'═'*56}\n")

    return results


# ══════════════════════════════════════════════
# 10. 메인
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="산업안전관리비 세금계산서 파싱 — PDF(pdfplumber) / 이미지(VLM) 자동 분기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python parse_tax_invoice.py --file 세금계산서.pdf
  python parse_tax_invoice.py --file 세금계산서.jpg
  python parse_tax_invoice.py --folder invoices/ --output ./results/
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   help="세금계산서 파일 경로 (.pdf / .jpg / .jpeg / .png / .tif / .tiff)")
    group.add_argument("--folder", help="세금계산서 파일 폴더 경로 (배치 처리)")

    parser.add_argument("--output",  default=None,
                        help="결과 JSON 저장 폴더 (기본: 입력 파일 폴더 내 tax_invoice_results/)")
    parser.add_argument("--secret",  default=None,
                        help="(deprecated) 과거 CLOVA OCR Secret — 더 이상 사용하지 않음")
    parser.add_argument("--url",     default=None,
                        help="(deprecated) 과거 CLOVA OCR URL — 더 이상 사용하지 않음")

    args = parser.parse_args()

    print(f"\n{'═'*56}")
    print(f"  세금계산서 파서 — 산업안전관리비 AI 검증 시스템")
    print(f"{'═'*56}")

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"[오류] 파일을 찾을 수 없습니다: {path}")
            return
        process_single(str(path), args.output, secret=args.secret, url=args.url)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"[오류] 폴더를 찾을 수 없습니다: {folder}")
            return
        process_folder(str(folder), args.output, secret=args.secret, url=args.url)


if __name__ == "__main__":
    main()
