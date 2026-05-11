"""
CLOVA OCR 영수증 특화모델 연동 모듈 v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 네이버 클로바 영수증 특화모델(Document OCR /receipt) 호출
- 응답 파싱 → 산업안전관리비 검증용 표준 JSON 출력
- 단일 이미지 / 배치(폴더) 처리 모두 지원

사용법:
    # 단일 이미지
    python clova_ocr_receipt.py --image 영수증.jpg

    # 폴더 내 전체 이미지 배치 처리
    python clova_ocr_receipt.py --folder ./증빙서류/ --output ./results/

    # API 키 직접 지정
    python clova_ocr_receipt.py --image 영수증.jpg --secret YOUR_SECRET_KEY --url YOUR_INVOKE_URL

설치:
    pip install requests Pillow python-dotenv

환경변수 설정 (.env 파일 또는 시스템 환경변수):
    CLOVA_OCR_SECRET=발급받은_Secret_Key
    CLOVA_OCR_URL=https://...apigw.ntruss.com/custom/v1/{DomainId}/{InvokeKey}/document/receipt

    .env.example 파일을 복사하여 .env 파일을 생성하고 키를 입력하세요.
"""

import os
import json
import uuid
import time
import base64
import argparse
import requests
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수 직접 설정 필요


# ══════════════════════════════════════════════
# 0. 설정 — API 키 / 엔드포인트 (.env 또는 환경변수에서 로드)
# ══════════════════════════════════════════════

CLOVA_SECRET  = os.environ.get("CLOVA_OCR_SECRET")
CLOVA_URL     = os.environ.get("CLOVA_OCR_URL")

# 신뢰도 임계값 — 이 값 미만의 필드는 null 처리
CONFIDENCE_THRESHOLD = 0.5

# 지원 이미지 확장자
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}


# ══════════════════════════════════════════════
# 1. CLOVA OCR API 호출
# ══════════════════════════════════════════════

def _build_url(url: str) -> str:
    """
    Invoke URL 끝에 /document/receipt 가 없으면 자동으로 붙여줌.
    예) .../8f7c18df...  →  .../8f7c18df.../document/receipt
    """
    url = url.rstrip("/")
    if not url.endswith("/document/receipt"):
        url = url + "/document/receipt"
    return url


def call_clova_receipt(image_path: str, secret: str, url: str) -> dict:
    """
    CLOVA OCR 영수증 특화모델 API 호출 (multipart/form-data 방식)

    Args:
        image_path: 영수증 이미지 경로 (jpg / png / pdf 등)
        secret:     X-OCR-SECRET 헤더값
        url:        Invoke URL (끝에 /document/receipt 없어도 자동 추가)

    Returns:
        CLOVA 원본 응답 dict (실패 시 {"error": "..."} 반환)
    """
    url = _build_url(url)
    ext = Path(image_path).suffix.lower().lstrip(".")
    fmt_map = {"jpg": "jpg", "jpeg": "jpg", "png": "png",
               "pdf": "pdf", "tif": "tif", "tiff": "tiff"}
    img_format = fmt_map.get(ext, "jpg")

    message = {
        "version":   "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),   # Unix timestamp (밀리초)
        "images": [
            {
                "format": img_format,
                "name":   Path(image_path).stem,
            }
        ]
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


# ══════════════════════════════════════════════
# 2. 응답 파싱 유틸리티
# ══════════════════════════════════════════════

def _get_value(obj: dict, threshold: float = CONFIDENCE_THRESHOLD) -> str | None:
    """
    CLOVA 공통 객체에서 정제값(formatted.value) 추출.
    신뢰도가 threshold 미만이면 None 반환.
    """
    if obj is None:
        return None
    score = obj.get("confidenceScore", 1.0)
    if score < threshold:
        return None
    # formatted.value 우선, 없으면 text 원본 반환
    formatted = obj.get("formatted")
    if formatted and formatted.get("value"):
        return formatted["value"]
    return obj.get("text")


def _get_date(date_obj: dict) -> str | None:
    """
    날짜 객체 → 'YYYY-MM-DD' 문자열.
    day=00 / month=00 등 유효하지 않은 날짜는 None 반환 (warnings에서 감지).
    """
    if date_obj is None:
        return None
    fmt = date_obj.get("formatted", {}) or {}
    y, m, d = fmt.get("year"), fmt.get("month"), fmt.get("day")
    if y and m and d:
        date_str = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            # "2025-04-00" 처럼 day=00 등 파이썬 datetime이 거부하는 값
            return None          # 호출부(validate_result)에서 경고 추가
    return date_obj.get("text")


def _get_time(time_obj: dict) -> str | None:
    """시간 객체 → 'HH:MM:SS' 문자열"""
    if time_obj is None:
        return None
    fmt = time_obj.get("formatted", {})
    h = fmt.get("hour",   "00")
    m = fmt.get("minute", "00")
    s = fmt.get("second", "00")
    return f"{h.zfill(2)}:{m.zfill(2)}:{s.zfill(2)}"


def _safe_int(value: str | None) -> int | None:
    """문자열 → 정수 변환.

    CLOVA OCR이 천단위 구분자를 쉼표(7,800) 대신 점(7.800)으로
    인식하는 경우가 있다. 한국 원화는 소수점이 없으므로 점도 제거한다.
    """
    if value is None:
        return None
    try:
        cleaned = (
            str(value)
            .replace(",", "")
            .replace(".", "")   # 천단위 구분 점 제거 (예: "15.800" → "15800")
            .replace(" ", "")
            .replace("원", "")
        )
        return int(cleaned)
    except ValueError:
        return None


# ══════════════════════════════════════════════
# 2-b. 거래명세표 품목 행 병합 유틸리티
# ══════════════════════════════════════════════

def _merge_split_items(items: list) -> list:
    """
    CLOVA 영수증 모델이 거래명세표를 처리할 때 발생하는 행 분리 문제 보정.

    현상:
      CLOVA가 품목명 행(name only)과 수량/금액 행(count/amount only)을
      각각 별개의 item으로 분리 추출하는 경우가 있음.
      예) {name: "안전발판(작업발판)", count: None, amount: None}
           {name: None,            count: 30,   amount: 66000}

    처리:
      - name이 있고 count·unit_price·amount 모두 없는 행 → 헤더 행으로 간주
      - 헤더 행 바로 다음에 name이 없고 count/amount가 있는 행이 연속되면 병합
      - 헤더 행 다음에 여러 수량행이 연속될 때도 첫 번째 수량행만 병합(나머지 유지)
    """
    if not items:
        return items

    merged = []
    i = 0
    while i < len(items):
        item = items[i]
        name       = item.get("name")
        count      = item.get("count")
        unit_price = item.get("unit_price")
        amount     = item.get("amount")

        # 품목명만 있고 수량/금액이 전부 없는 행 → 병합 시도
        if name and count is None and unit_price is None and amount is None:
            if i + 1 < len(items):
                nxt = items[i + 1]
                if (not nxt.get("name")) and (nxt.get("count") or nxt.get("amount")):
                    merged.append({
                        "name":       name,
                        "count":      nxt.get("count"),
                        "unit_price": nxt.get("unit_price"),
                        "amount":     nxt.get("amount"),
                    })
                    i += 2
                    continue

        merged.append(item)
        i += 1

    return merged


# ══════════════════════════════════════════════
# 3. CLOVA 응답 → 표준 JSON 변환
# ══════════════════════════════════════════════

def parse_clova_response(raw: dict) -> dict:
    """
    CLOVA OCR 영수증 원본 응답을 산업안전관리비 검증용 표준 JSON으로 변환.

    표준 JSON 구조:
    {
      "ocr_type":      "receipt",
      "source_file":   "영수증.jpg",
      "ocr_engine":    "clova_receipt_v2",
      "infer_result":  "SUCCESS" | "FAILURE" | "ERROR",
      "estimated_language": "ko",
      "store": {
        "name":    "점포명",
        "biz_num": "사업자등록번호",
        "address": "주소",
        "tel":     "전화번호"
      },
      "payment": {
        "date":        "YYYY-MM-DD",
        "time":        "HH:MM:SS",
        "card_company": "카드사명",
        "card_number":  "카드번호",
        "confirm_num":  "승인번호"
      },
      "items": [
        {
          "name":       "상품명",
          "count":      수량(int),
          "unit_price": 단가(int),
          "amount":     합계금액(int)
        }
      ],
      "total_amount":  총액(int),
      "tax_amount":    부가세(int),
      "discount_amount": 할인금액(int),
      "raw_clova":     {...}   # CLOVA 원본 응답 (디버깅용)
    }
    """

    # ── 오류 처리 ──────────────────────────────────
    if "error" in raw:
        return {
            "ocr_type":     "receipt",
            "infer_result": "ERROR",
            "error":        raw["error"],
            "raw_clova":    raw,
        }

    images = raw.get("images", [])
    if not images:
        return {
            "ocr_type":     "receipt",
            "infer_result": "ERROR",
            "error":        "응답에 images 배열 없음",
            "raw_clova":    raw,
        }

    img = images[0]
    infer_result = img.get("inferResult", "ERROR")
    source_name  = img.get("name", "")

    base_result = {
        "ocr_type":          "receipt",
        "source_file":       source_name,
        "ocr_engine":        "clova_receipt_v2",
        "infer_result":      infer_result,
        "estimated_language": img.get("receipt", {}).get("meta", {}).get("estimatedLanguage"),
        "store":             {},
        "payment":           {},
        "items":             [],
        "total_amount":      None,
        "tax_amount":        None,
        "discount_amount":   None,
        "raw_clova":         raw,
    }

    if infer_result != "SUCCESS":
        base_result["error"] = img.get("message", "인식 실패")
        return base_result

    result = img.get("receipt", {}).get("result", {})

    # ── 점포 정보 ───────────────────────────────────
    store_info = result.get("storeInfo", {})
    # CLOVA 영수증 모델은 상호명을 name / subName 중 하나에만 담는 경우가 있음.
    # subName은 신뢰도가 임계값(0.5) 미만이어도 매장명으로 쓸 수 있으므로
    # raw text를 fallback으로 사용한다 (예: 광화문점 confidence=0.44).
    sub_name_raw = (store_info.get("subName") or {}).get("text")
    store_name = (
        _get_value(store_info.get("name"))
        or _get_value(store_info.get("subName"))
        or sub_name_raw                          # 신뢰도 미달이어도 텍스트 사용
        or _get_value(store_info.get("bizNum"))
    )
    base_result["store"] = {
        "name":     store_name,
        "sub_name": _get_value(store_info.get("subName")),
        "biz_num":  _get_value(store_info.get("bizNum")),
        "address": _get_value(
            store_info.get("addresses", [{}])[0]
            if store_info.get("addresses") else None
        ),
        "tel": _get_value(
            store_info.get("tel", [{}])[0]
            if store_info.get("tel") else None
        ),
    }

    # ── 결제 정보 ───────────────────────────────────
    pay = result.get("paymentInfo", {})
    card = pay.get("cardInfo", {})
    base_result["payment"] = {
        "date":         _get_date(pay.get("date")),
        "time":         _get_time(pay.get("time")),
        "card_company": _get_value(card.get("company")),
        "card_number":  _get_value(card.get("number")),
        "confirm_num":  _get_value(card.get("confirmNum")),
    }

    # ── 상품 목록 ───────────────────────────────────
    import logging as _logging
    _log = _logging.getLogger(__name__)
    _log.debug("[CLOVA raw subResults] %s", __import__("json").dumps(result.get("subResults", []), ensure_ascii=False))
    _log.debug("[CLOVA raw storeInfo] %s", __import__("json").dumps(result.get("storeInfo", {}), ensure_ascii=False))

    items = []
    for group in result.get("subResults", []):
        for item in group.get("items", []):
            price_obj = item.get("price", {}) or {}
            # CLOVA 영수증 모델은 POS 영수증과 일반 영수증 형식에 따라
            # 금액을 price.price 또는 price.unitPrice 중 하나에만 담는다.
            # 둘 다 시도해 먼저 유효한 값을 사용한다.
            raw_amount     = _safe_int(_get_value(price_obj.get("price")))
            raw_unit_price = _safe_int(_get_value(price_obj.get("unitPrice")))
            count = _safe_int(_get_value(item.get("count")))

            # amount 확정 로직:
            #   1) price.price 가 있으면 그대로 사용
            #   2) 없고 unitPrice × count 계산 가능하면 계산값 사용
            #   3) unitPrice만 있고 count 없거나 1이면 unitPrice를 amount로 사용
            if raw_amount:
                amount = raw_amount
            elif raw_unit_price and count:
                amount = raw_unit_price * count
            elif raw_unit_price:
                amount = raw_unit_price  # count 미인식 → unitPrice를 합계로 간주
            else:
                amount = None

            items.append({
                "name":       _get_value(item.get("name")),
                "count":      count,
                "unit_price": raw_unit_price,
                "amount":     amount,
            })
    # 거래명세표 행 분리 현상 보정 (품목명 행 + 수량/금액 행 병합)
    base_result["items"] = _merge_split_items(items)

    # ── 합계 금액 ───────────────────────────────────
    total_price = result.get("totalPrice", {})
    base_result["total_amount"] = _safe_int(_get_value(total_price.get("price")))

    # ── 부가세 / 할인 ────────────────────────────────
    for sub in result.get("subTotal", []):
        tax_list = sub.get("taxPrice", [])
        disc_list = sub.get("discountPrice", [])

        if tax_list:
            tax_vals = [_safe_int(_get_value(t)) for t in tax_list if _get_value(t)]
            base_result["tax_amount"] = sum(v for v in tax_vals if v)

        if disc_list:
            disc_vals = [_safe_int(_get_value(d)) for d in disc_list if _get_value(d)]
            base_result["discount_amount"] = sum(v for v in disc_vals if v)

    return base_result


# ══════════════════════════════════════════════
# 4. 후처리 검증 로직
# ══════════════════════════════════════════════

def validate_result(parsed: dict) -> dict:
    """
    파싱 결과 후처리 검증:
    1. 품목 금액 합산 vs 총액 일치 여부
    2. 총액 = 공급가액 + 부가세 검증
    3. 필수 필드 존재 여부 확인

    Returns:
        validation_flags dict (원본 parsed에 "validation" 키로 추가됨)
    """
    flags = {
        "items_sum_match":  None,   # 품목 합산 == total_amount
        "tax_calc_match":   None,   # total - tax == net_amount (10% VAT 기준)
        "has_required_fields": False,
        "missing_fields":  [],
        "warnings":        [],      # 경고: 통과하되 검토 권장 (반올림 등 경미한 차이)
        "errors":          [],      # 오류: 반드시 검토 필요 (명백한 금액 불일치)
    }

    # ── 필수 필드 점검 ──────────────────────────────
    required = ["store.biz_num", "payment.date", "total_amount"]
    missing = []
    for field in required:
        keys = field.split(".")
        val = parsed
        for k in keys:
            val = val.get(k) if isinstance(val, dict) else None
        if val is None:
            missing.append(field)

    flags["missing_fields"]      = missing
    flags["has_required_fields"] = (len(missing) == 0)

    if missing:
        flags["warnings"].append(f"필수 필드 누락: {', '.join(missing)}")

    # ── 품목 합산 검증 ──────────────────────────────
    # 판정 기준:
    #   ±1원 이내  → 정상 ✅
    #   ±2~10원   → 반올림 의심 경고 ⚠️  (통과, warnings에 기록)
    #   10원 초과  → 금액 불일치 오류 ❌  (실패, errors에 기록)
    total = parsed.get("total_amount")
    tax   = parsed.get("tax_amount") or 0
    items = parsed.get("items", [])
    if total and items:
        items_sum = sum(item.get("amount") or 0 for item in items)
        if items_sum > 0:
            diff_direct   = abs(total - items_sum)
            diff_with_tax = abs(total - items_sum - tax)
            best_diff     = min(diff_direct, diff_with_tax)

            if best_diff <= 1:
                # 정상
                flags["items_sum_match"] = True

            elif best_diff <= 10:
                # 반올림 오차 의심 — 통과하되 경고
                flags["items_sum_match"] = True
                flags["warnings"].append(
                    f"품목 합산 반올림 의심 ({best_diff}원 차이): "
                    f"품목합산 {items_sum:,}원, 총액 {total:,}원 — 검토 권장"
                )

            else:
                # 명백한 금액 불일치 — 오류
                flags["items_sum_match"] = False
                flags["errors"].append(
                    f"품목 합산 불일치: 품목합산({items_sum:,}원) ≠ 총액({total:,}원) "
                    f"— 차이 {diff_direct:,}원, 반드시 확인 필요"
                )

    # ── 부가세 검증 (10% VAT) ───────────────────────
    if total and tax:
        expected_tax = round(total / 11)   # 총액에서 역산한 부가세
        diff = abs(tax - expected_tax)
        flags["tax_calc_match"] = (diff <= 10)   # 10원 이내 오차 허용
        if diff > 10:
            flags["warnings"].append(
                f"부가세 불일치: 인식값={tax:,}원, 역산값={expected_tax:,}원"
            )

    # ── 단가 × 수량 = 합계 검증 ─────────────────────
    for item in parsed.get("items", []):
        iname      = item.get("name") or "미상"
        unit_price = item.get("unit_price")
        amount     = item.get("amount")
        count      = item.get("count")
        if unit_price and amount and count:
            expected_amount = unit_price * count
            if expected_amount != amount:
                flags["warnings"].append(
                    f"단가×수량≠합계: '{iname}' "
                    f"단가({unit_price:,}원) × 수량({count}개) = {expected_amount:,}원 ≠ 합계({amount:,}원)"
                )

    # ── 날짜 유효성 이상값 탐지 ─────────────────────
    date_val = parsed.get("payment", {}).get("date")
    if date_val is None and parsed.get("payment", {}).get("date") is not None:
        # _get_date가 None을 반환한 경우(day=00 등)는 payment.date 자체가 None
        pass  # payment.date가 None이면 이미 missing_fields에 포함됨

    # raw에서 날짜 텍스트를 재확인하여 0일/0월 탐지
    try:
        raw_images = parsed.get("raw_clova", {}).get("images", [{}])
        raw_date_fmt = (
            raw_images[0]
            .get("receipt", {})
            .get("result", {})
            .get("paymentInfo", {})
            .get("date", {})
            .get("formatted", {})
        ) or {}
        raw_day = raw_date_fmt.get("day", "")
        raw_mon = raw_date_fmt.get("month", "")
        if raw_day in ("0", "00") or raw_mon in ("0", "00"):
            flags["warnings"].append(
                f"날짜 이상값: day={raw_day}, month={raw_mon} — 유효하지 않은 날짜로 무효 처리됨"
            )
    except Exception:
        pass

    # ── 간이영수증 / 불완전 인식 감지 ────────────────
    if not parsed.get("items") and not parsed.get("payment", {}).get("date"):
        flags["warnings"].append(
            "간이영수증 또는 불완전 인식 의심 — 품목 목록 없음, 결제일 없음"
        )

    parsed["validation"] = flags
    return parsed


# ══════════════════════════════════════════════
# 5. 파일 저장 유틸리티
# ══════════════════════════════════════════════

def save_result(parsed: dict, output_path: str):
    """파싱 결과를 JSON 파일로 저장 (raw_clova 제외 버전 + 전체 버전 분리 저장)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 표준 JSON (raw 제외, 다운스트림 에이전트용) ──
    clean = {k: v for k, v in parsed.items() if k != "raw_clova"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    # ── 전체 원본 포함 버전 (디버그용) ──────────────
    raw_path = output_path.with_stem(output_path.stem + "_raw")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return str(output_path)


# ══════════════════════════════════════════════
# 6. 콘솔 요약 출력
# ══════════════════════════════════════════════

def print_summary(parsed: dict):
    """터미널에 파싱 결과 요약 출력"""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  📄 파일:       {parsed.get('source_file', 'N/A')}")
    print(f"  🔍 인식 결과:  {parsed.get('infer_result', 'N/A')}")
    print(f"{sep}")

    if parsed.get("infer_result") != "SUCCESS":
        print(f"  ⚠️  오류: {parsed.get('error', '알 수 없는 오류')}")
        return

    store = parsed.get("store", {})
    pay   = parsed.get("payment", {})

    print(f"  🏪 점포명:     {store.get('name') or '─'}")
    print(f"  📋 사업자번호: {store.get('biz_num') or '─'}")
    print(f"  📍 주소:       {store.get('address') or '─'}")
    print(f"  📞 전화:       {store.get('tel') or '─'}")
    print(f"{sep}")
    print(f"  📅 결제일:     {pay.get('date') or '─'}  {pay.get('time') or ''}")
    print(f"  💳 카드사:     {pay.get('card_company') or '─'}")
    print(f"  🔑 승인번호:   {pay.get('confirm_num') or '─'}")
    print(f"{sep}")

    items = parsed.get("items", [])
    if items:
        print(f"  🛒 품목 ({len(items)}개):")
        for item in items:
            name   = item.get("name") or "─"
            count  = item.get("count")
            amount = item.get("amount")
            count_str  = f"×{count}"  if count  else ""
            amount_str = f"{amount:,}원" if amount else "─"
            print(f"     · {name} {count_str}  →  {amount_str}")
    else:
        print(f"  🛒 품목:       인식된 항목 없음")

    print(f"{sep}")
    total    = parsed.get("total_amount")
    tax      = parsed.get("tax_amount")
    discount = parsed.get("discount_amount")
    print(f"  💰 총액:       {f'{total:,}원' if total else '─'}")
    if tax:      print(f"  🧾 부가세:     {tax:,}원")
    if discount: print(f"  🏷️  할인:       {discount:,}원")

    # ── 검증 결과 ──────────────────────────────────
    val = parsed.get("validation", {})
    if val:
        print(f"{sep}")
        print(f"  ✅ 필수 필드:  {'완전' if val.get('has_required_fields') else '불완전 ⚠️'}")
        if val.get("items_sum_match") is not None:
            match_str = "일치 ✅" if val["items_sum_match"] else "불일치 ❌"
            print(f"  🔢 금액 합산: {match_str}")
        for w in val.get("warnings", []):
            print(f"  ⚠️  {w}")

    print(f"{sep}\n")


# ══════════════════════════════════════════════
# 7. 단일 / 배치 처리
# ══════════════════════════════════════════════

def process_single(image_path: str, secret: str, url: str,
                   output_dir: str = None) -> dict:
    """단일 이미지 처리 → 파싱 → 검증 → 저장"""
    image_path = Path(image_path)
    print(f"\n처리 중: {image_path.name}")

    # 1) API 호출
    raw = call_clova_receipt(str(image_path), secret, url)

    # 2) 파싱
    parsed = parse_clova_response(raw)
    parsed["source_file"] = image_path.name   # 파일명 보정

    # 3) 후처리 검증
    parsed = validate_result(parsed)

    # 4) 저장
    out_dir  = Path(output_dir) if output_dir else image_path.parent / "ocr_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"{image_path.stem}_ocr_{timestamp}.json"
    saved     = save_result(parsed, str(out_path))

    # 5) 콘솔 요약
    print_summary(parsed)
    print(f"  💾 저장: {saved}")

    return parsed


def process_folder(folder_path: str, secret: str, url: str,
                   output_dir: str = None) -> list[dict]:
    """폴더 내 지원 이미지 전체 배치 처리"""
    folder = Path(folder_path)
    images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print(f"[경고] 지원 이미지 파일 없음: {folder}")
        return []

    out_dir = Path(output_dir) if output_dir else folder / "ocr_results"
    print(f"\n📂 배치 처리: {folder.name}  ({len(images)}개 파일)")
    print(f"📁 결과 저장: {out_dir}\n")

    results = []
    for i, img in enumerate(sorted(images), 1):
        print(f"[{i}/{len(images)}]", end=" ")
        result = process_single(str(img), secret, url, str(out_dir))
        results.append(result)
        time.sleep(0.3)   # API 호출 간격 (과부하 방지)

    # ── 배치 요약 ──────────────────────────────────
    success = sum(1 for r in results if r.get("infer_result") == "SUCCESS")
    fail    = len(results) - success
    print(f"\n{'═'*52}")
    print(f"  배치 처리 완료: 총 {len(results)}개 | ✅ 성공 {success}개 | ❌ 실패 {fail}개")
    print(f"{'═'*52}\n")

    return results


# ══════════════════════════════════════════════
# 8. 메인
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLOVA OCR 영수증 특화모델 연동 — 산업안전관리비 검증용"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="단일 영수증 이미지 경로")
    group.add_argument("--folder", help="영수증 이미지 폴더 경로 (배치 처리)")

    parser.add_argument("--output", default=None,
                        help="결과 JSON 저장 폴더 (기본: 이미지 폴더 내 ocr_results/)")
    parser.add_argument("--secret", default=None,
                        help="X-OCR-SECRET 키 (미입력 시 환경변수 CLOVA_OCR_SECRET 사용)")
    parser.add_argument("--url",    default=None,
                        help="CLOVA OCR 엔드포인트 URL (미입력 시 환경변수 CLOVA_OCR_URL 사용)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="신뢰도 임계값 (기본: 0.5, 범위: 0.0~1.0)")

    args = parser.parse_args()

    # ── API 키 / URL 결정 ───────────────────────────
    secret = args.secret or CLOVA_SECRET
    url    = args.url    or CLOVA_URL

    if not secret:
        print("[오류] CLOVA OCR Secret Key가 없습니다.")
        print("       --secret 옵션 또는 환경변수 CLOVA_OCR_SECRET을 설정해주세요.")
        return

    if not url:
        print("[오류] CLOVA OCR URL이 없습니다.")
        print("       --url 옵션 또는 환경변수 CLOVA_OCR_URL을 설정해주세요.")
        return

    # 신뢰도 임계값 적용
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    print(f"\n{'═'*52}")
    print(f"  CLOVA OCR 영수증 특화모델")
    print(f"  신뢰도 임계값: {CONFIDENCE_THRESHOLD}")
    print(f"{'═'*52}")

    # ── 처리 실행 ───────────────────────────────────
    if args.image:
        if not Path(args.image).exists():
            print(f"[오류] 이미지 파일을 찾을 수 없습니다: {args.image}")
            return
        process_single(args.image, secret, url, args.output)

    elif args.folder:
        if not Path(args.folder).is_dir():
            print(f"[오류] 폴더를 찾을 수 없습니다: {args.folder}")
            return
        process_folder(args.folder, secret, url, args.output)


if __name__ == "__main__":
    main()
