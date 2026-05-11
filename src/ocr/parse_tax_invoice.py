"""
산업안전관리비 세금계산서 파싱 모듈 v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파일 확장자를 자동 감지해 처리 방식을 분기한다.
  - .pdf              → pdfplumber 텍스트 직접 추출 (parse_usage_statement.py 동일 방식)
  - .jpg/.jpeg/.png/.tif/.tiff → CLOVA OCR API 호출  (clova_ocr_receipt.py 동일 방식)

두 경로 모두 처리 후 동일한 JSON 구조로 출력.

사용법:
    # 단일 파일 (PDF 또는 이미지)
    python parse_tax_invoice.py --file 세금계산서.pdf
    python parse_tax_invoice.py --file 세금계산서.jpg

    # 폴더 배치 처리
    python parse_tax_invoice.py --folder invoices/

    # 출력 폴더 지정
    python parse_tax_invoice.py --file 세금계산서.pdf --output ./results/

    # API 키 직접 지정 (이미지 경로일 때만 사용)
    python parse_tax_invoice.py --file 세금계산서.jpg --secret YOUR_KEY --url YOUR_URL

설치:
    pip install pdfplumber requests python-dotenv

환경변수 (이미지 처리 시 필요, .env 파일 또는 시스템 환경변수):
    CLOVA_OCR_SECRET=발급받은_Secret_Key
    CLOVA_OCR_URL=https://...apigw.ntruss.com/custom/v1/.../document/receipt

    .env.example 파일을 복사하여 .env 파일을 생성하고 키를 입력하세요.
"""

import os
import re
import json
import uuid
import time
import argparse
import requests
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
# 0. 설정 — API 키 / 엔드포인트 (.env 또는 환경변수에서 로드)
# ══════════════════════════════════════════════

CLOVA_SECRET = os.environ.get("CLOVA_OCR_SECRET")
CLOVA_URL    = os.environ.get("CLOVA_OCR_URL")

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
# 4. 이미지 경로 — CLOVA OCR API 호출
# ══════════════════════════════════════════════

def _call_clova_ocr(image_path: str, secret: str, url: str) -> dict:
    """
    CLOVA OCR API 호출 (clova_ocr_receipt.py와 동일 방식, multipart/form-data).

    Returns:
        CLOVA 원본 응답 dict (실패 시 {"error": "..."} 반환)
    """
    url = url.rstrip("/")
    if not url.endswith("/document/receipt"):
        url = url + "/document/receipt"

    ext = Path(image_path).suffix.lower().lstrip(".")
    fmt_map = {"jpg": "jpg", "jpeg": "jpg", "png": "png",
               "tif": "tif", "tiff": "tiff"}
    img_format = fmt_map.get(ext, "jpg")

    message = {
        "version":   "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [
            {
                "format": img_format,
                "name":   Path(image_path).stem,
            }
        ],
    }

    headers = {"X-OCR-SECRET": secret}

    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data={"message": json.dumps(message)},
                files={"file": (Path(image_path).name, f, f"image/{img_format}")},
                timeout=30,
            )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP 오류 {response.status_code}: {response.text[:300]}"}
    except requests.exceptions.Timeout:
        return {"error": "요청 시간 초과 (30초)"}
    except Exception as e:
        return {"error": str(e)}


def _extract_raw_text_from_clova(raw: dict) -> str:
    """
    CLOVA OCR 원본 응답에서 인식된 전체 텍스트를 추출.
    receipt 구조화 응답과 일반 fields 응답 모두 처리.
    """
    if "error" in raw:
        return ""

    images = raw.get("images", [])
    if not images:
        return ""

    img = images[0]
    texts = []

    # 방법 1: receipt 구조화 결과에서 텍스트 수집
    receipt = img.get("receipt", {}).get("result", {})
    if receipt:
        # storeInfo → name
        store = receipt.get("storeInfo", {})
        if store.get("name", {}).get("text"):
            texts.append(store["name"]["text"])

        # subResults → items
        for sub in receipt.get("subResults", []):
            for item in sub.get("items", []):
                name_obj = item.get("name", {})
                if name_obj and name_obj.get("text"):
                    texts.append(name_obj["text"])
                price_obj = item.get("price", {})
                for k in ["unitPrice", "price"]:
                    if price_obj.get(k, {}).get("text"):
                        texts.append(price_obj[k]["text"])

        # totalPrice
        tp = receipt.get("totalPrice", {}).get("price", {})
        if tp.get("text"):
            texts.append(tp["text"])

    # 방법 2: fields 배열에서 직접 텍스트 수집 (일반 OCR 응답)
    for field in img.get("fields", []):
        t = field.get("inferText", "") or field.get("text", "")
        if t.strip():
            texts.append(t.strip())

    return "\n".join(texts)


def parse_from_image(file_path: str, secret: str, url: str) -> dict:
    """
    이미지 세금계산서 파싱 (CLOVA OCR 사용).

    Args:
        file_path: 이미지 파일 경로
        secret:    CLOVA OCR Secret Key
        url:       CLOVA OCR Invoke URL

    Returns:
        표준 세금계산서 JSON dict
    """
    result = _empty_result(file_path, "image", "ocr")

    # CLOVA OCR API 호출
    raw = _call_clova_ocr(str(file_path), secret, url)

    if "error" in raw:
        result["validation"]["warnings"].append(f"OCR 오류: {raw['error']}")
        return _validate(result)

    images = raw.get("images", [])
    if not images or images[0].get("inferResult") != "SUCCESS":
        msg = images[0].get("message", "OCR 인식 실패") if images else "응답 없음"
        result["validation"]["warnings"].append(f"OCR 실패: {msg}")
        return _validate(result)

    # 원본 텍스트 추출
    raw_text = _extract_raw_text_from_clova(raw)
    result["raw_text"] = raw_text

    # 필드 추출 (텍스트 파싱 공통 로직)
    extracted = _extract_fields_from_text(raw_text)
    result["supplier"]            = extracted["supplier"]
    result["buyer"]               = extracted["buyer"]
    result["issue_date"]          = extracted["issue_date"]
    result["items"]               = extracted["items"]
    result["total_supply_amount"] = extracted["total_supply_amount"]
    result["total_tax_amount"]    = extracted["total_tax_amount"]
    result["total_amount"]        = extracted["total_amount"]

    result = _validate(result)
    return result


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

    확장자를 자동 감지해 PDF / 이미지 경로를 분기.

    Args:
        file_path: 세금계산서 파일 경로 (.pdf / .jpg / .jpeg / .png / .tif / .tiff)
        secret:    CLOVA OCR Secret Key (이미지 처리 시 필요)
        url:       CLOVA OCR Invoke URL  (이미지 처리 시 필요)

    Returns:
        표준 JSON dict
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext in PDF_EXTS:
        return parse_from_pdf(str(path))

    elif ext in IMAGE_EXTS:
        _secret = secret or CLOVA_SECRET
        _url    = url    or CLOVA_URL
        if not _secret:
            result = _empty_result(file_path, "image", "ocr")
            result["validation"]["warnings"].append(
                "CLOVA OCR Secret Key 없음 — --secret 옵션 또는 환경변수 CLOVA_OCR_SECRET 설정 필요"
            )
            return result
        return parse_from_image(str(path), _secret, _url)

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
        description="산업안전관리비 세금계산서 파싱 — PDF(pdfplumber) / 이미지(CLOVA OCR) 자동 분기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python parse_tax_invoice.py --file 세금계산서.pdf
  python parse_tax_invoice.py --file 세금계산서.jpg --secret KEY --url URL
  python parse_tax_invoice.py --folder invoices/ --output ./results/
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   help="세금계산서 파일 경로 (.pdf / .jpg / .jpeg / .png / .tif / .tiff)")
    group.add_argument("--folder", help="세금계산서 파일 폴더 경로 (배치 처리)")

    parser.add_argument("--output",  default=None,
                        help="결과 JSON 저장 폴더 (기본: 입력 파일 폴더 내 tax_invoice_results/)")
    parser.add_argument("--secret",  default=None,
                        help="CLOVA OCR Secret Key (이미지 처리 시; 미입력 시 환경변수 CLOVA_OCR_SECRET)")
    parser.add_argument("--url",     default=None,
                        help="CLOVA OCR Invoke URL (이미지 처리 시; 미입력 시 환경변수 CLOVA_OCR_URL)")

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
