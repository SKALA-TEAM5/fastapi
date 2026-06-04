"""
VLM OCR — Vision Language Model 기반 증빙서류 인식 모듈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
산업안전관리비 AI 검증 시스템 — src/ocr/vlm_ocr.py

현재 기본 모델: Google Gemini (gemini-3.1-flash-lite)
VLM_PROVIDER=openai 로 OpenAI GPT-4o 계열 전환 가능

[핵심 설계 원칙]
  - parse_vision_response() 출력 스키마 = parse_clova_response() 출력 스키마
  - receipt_validator.validate_result() 재사용 → 다운스트림 매칭 엔진 변경 불필요
  - VLM_PROVIDER 환경변수로 Gemini/OpenAI 전환 (코드 변경 불필요)

[사용법]
    from src.ocr.vlm_ocr import parse_vision_response

    result = parse_vision_response("영수증.jpg", type_hint="receipt")

[환경변수]
    VLM_PROVIDER=gemini               # "gemini" | "openai"

    # Gemini (기본값)
    GEMINI_API_KEY=AIza...
    GEMINI_MODEL=gemini-3.1-flash-lite
    GEMINI_MODEL_FALLBACK=gemini-2.5-flash

    # OpenAI (VLM_PROVIDER=openai 전환 시)
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_MODEL_FALLBACK=gpt-4o

[주의]
  현장사진 분석용 Vision Agent(orchestrator의 vision 에이전트)와 별개 모듈.
  이 파일은 영수증·거래명세표 등 증빙서류 OCR 전용이다.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

# validate_result — CLOVA 의존성 없는 독립 검증 모듈
from src.ocr.receipt_validator import validate_result

try:
    from src.core.config import (
        VLM_PROVIDER,
        GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODEL_FALLBACK,
        OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MODEL_FALLBACK,
    )
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    VLM_PROVIDER          = os.getenv("VLM_PROVIDER", "gemini")
    GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash")
    OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")


# ══════════════════════════════════════════════
# 1. VLM 프롬프트 (Gemini / OpenAI 공통)
# ══════════════════════════════════════════════

_SYSTEM_PROMPT = """\
당신은 건설업 안전관리비 증빙서류를 분석하는 전문 AI입니다.
이미지를 분석하여 반드시 아래의 JSON 형식으로만 응답하세요.
코드 블록(```json)이나 부가 설명 없이 JSON 객체만 출력하세요.
"""

_USER_PROMPT_TEMPLATE = """\
이미지를 분석하여 다음 JSON 스키마로 정보를 추출해주세요.

사용자가 선택한 문서 유형 힌트: {type_hint}

추출 JSON 스키마:
{{
  "doc_type": "receipt | trade_statement | tax_invoice | site_photo | unknown",
  "infer_result": "SUCCESS | PARTIAL | FAILED",
  "store": {{
    "name": "업체명 또는 공급자명",
    "biz_num": "사업자등록번호 (000-00-00000 형식)",
    "address": "주소",
    "tel": "전화번호"
  }},
  "payment": {{
    "date": "YYYY-MM-DD 또는 null",
    "date_status": "recognized | not_written | unreadable",
    "time": "HH:MM:SS 또는 null",
    "card_company": "카드사명 또는 null",
    "card_number": "카드번호 마지막 4자리 또는 null",
    "confirm_num": "승인번호 또는 null"
  }},
  "items": [
    {{
      "name": "품목명",
      "count": 수량(정수) 또는 null,
      "unit_price": 단가(정수, 원 단위) 또는 null,
      "amount": 금액(정수, 원 단위) 또는 null
    }}
  ],
  "total_amount": 합계금액(정수, 원 단위) 또는 null,
  "tax_amount": 부가세(정수, 원 단위) 또는 null,
  "discount_amount": 할인금액(정수, 원 단위) 또는 null,
  "confidence": 0.0~1.0 사이 신뢰도,
  "fail_reason": "FAILED 시 실패 사유, 그 외 null"
}}

── 기본 추출 규칙 ──────────────────────────────────────
- 금액은 숫자만 추출 (원, 쉼표, 공백 제거). 예: "15,800원" → 15800
- 날짜는 이미지 전체에서 연·월·일 정보를 적극적으로 탐색한다
  · 직인(도장) 안의 날짜, 상단·하단 모서리의 날짜 포함
  · "25. 3. 15" / "2025.03.15" / "25/03/15" 등 다양한 형식 → YYYY-MM-DD 변환
  · 연도가 두 자리("25")면 2000년대("2025")로 해석
  · 끝내 날짜를 찾을 수 없을 때만 null
- 날짜는 이미지에서 실제로 확인된 부분만 기입한다 (절대 추측·보완 금지)
  · 연도만 있고 월·일 없음 → "YYYY-01-01" 등으로 채우지 말고 null 반환
  · 연·월만 있고 일 없음   → null 반환
  · 월·일만 있고 연도 없음 → null 반환
  · 연·월·일 모두 확인된 경우에만 YYYY-MM-DD 형식으로 반환
- date_status 판정 기준 (date 값과 반드시 쌍으로 반환):
  · 날짜를 정상 추출했으면               → "recognized"  (date에 값 있음)
  · 영수증에 날짜 기입란/날짜 자체가 없음 → "not_written" (date = null)
  · 날짜가 있어 보이나 판독 불가          → "unreadable"  (date = null)
- 품목이 없으면 items를 빈 배열 []로 반환
- infer_result 판정:
    SUCCESS  = 업체명·날짜·금액 모두 추출 성공
    PARTIAL  = 일부 필드만 추출 (날짜 없음, 업체명 불확실 등)
    FAILED   = 판독 불가 (이미지 품질 불량, 관련 없는 파일 등)

── 수기 문서 특별 규칙 ────────────────────────────────
수기(손글씨)가 포함된 문서는 아래 규칙을 반드시 적용한다.

[숫자 혼동 방지]
- "1"과 "7" 구분: 획 상단에 가로 획(﹁)이 있으면 "7", 없으면 "1"
- "0"과 "6" 구분: 닫힌 원이면 "0", 위가 열린 곡선이면 "6"
- "1"과 "7" 구분이 애매한 경우, 금액 문맥(합계·부가세·단가와의 정합성)을 함께 고려한다
  예) 단가 1,700원 × 수량 10 = 합계 17,000원 → 7 가능성 검토

[금액 정합성 검증 — 반드시 수행]
1. 품목별 (단가 × 수량) 계산값과 합계 금액이 일치하는지 확인한다
2. 품목 금액 합산이 total_amount와 일치하는지 확인한다
3. 불일치 시: 각 숫자를 혼동 가능 숫자(1↔7, 0↔6 등)로 교체해 재계산한다
4. 정합성이 맞는 해석을 최종값으로 채택한다

[날짜 탐색 — 수기 특화]
- 직인(도장) 속 날짜, 서명 옆 날짜, 영수증 상·하단 손글씨 날짜를 모두 확인한다
- 날짜처럼 보이는 숫자 조합(YY.MM.DD, MM월 DD일 등)은 모두 날짜 후보로 검토한다

[confidence 기준]
- 모든 필드가 확실하면 0.90 이상
- 숫자 혼동 의심이 있었으나 정합성으로 해결했으면 0.75~0.89
- 날짜 또는 금액이 불확실하면 0.74 이하
"""


def _build_user_prompt(type_hint: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(type_hint=type_hint)


# ══════════════════════════════════════════════
# 2. 공통 유틸리티
# ══════════════════════════════════════════════

def _parse_json_from_text(raw_text: str) -> dict | None:
    """LLM 응답 문자열에서 JSON 추출. 실패 시 None 반환."""
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 코드 블록 제거 후 재시도
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════
# 3-A. Gemini 호출
# ══════════════════════════════════════════════

_gemini_client: Optional[object] = None


def _get_gemini_client():
    """google.genai 클라이언트 싱글턴 반환"""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 GEMINI_API_KEY=AIza... 를 추가해주세요."
            )
        from google import genai
        _gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1"},
        )
    return _gemini_client


def call_vision_gemini(
    image_path: str,
    type_hint: str = "unknown",
    model: str | None = None,
) -> dict:
    """
    Google Gemini Vision API 호출 (google.genai 신규 SDK 사용).

    Args:
        image_path: 로컬 이미지 경로
        type_hint:  사용자가 선택한 문서 유형 힌트
        model:      사용할 모델 (None이면 config.GEMINI_MODEL)

    Returns:
        VLM이 반환한 JSON dict (실패 시 {"error": "..."} 반환)
    """
    from PIL import Image

    _model = model or GEMINI_MODEL

    try:
        client = _get_gemini_client()
    except EnvironmentError as e:
        return {"error": str(e)}

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return {"error": f"이미지 파일을 찾을 수 없습니다: {image_path}"}
    except Exception as e:
        return {"error": f"이미지 로드 실패: {e}"}

    prompt = _SYSTEM_PROMPT + "\n\n" + _build_user_prompt(type_hint)

    try:
        response = client.models.generate_content(
            model=_model,
            contents=[img, prompt],
        )
        raw_text = response.text

    except Exception as e:
        return {"error": f"Gemini API 호출 실패: {e}", "model_used": _model}

    parsed_json = _parse_json_from_text(raw_text)
    if parsed_json is None:
        return {
            "error": f"Gemini 응답 JSON 파싱 실패: {raw_text[:300]}",
            "model_used": _model,
        }

    parsed_json["model_used"] = _model
    return parsed_json


# ══════════════════════════════════════════════
# 3-B. OpenAI 호출 (VLM_PROVIDER=openai 시 사용)
# ══════════════════════════════════════════════

_openai_client: Optional[object] = None


def _get_openai_client():
    """OpenAI 클라이언트 싱글턴 반환"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY=sk-... 를 추가해주세요."
            )
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def call_vision_openai(
    image_path: str,
    type_hint: str = "unknown",
    model: str | None = None,
) -> dict:
    """
    OpenAI GPT-4o Vision API 호출.

    Args:
        image_path: 로컬 이미지 경로
        type_hint:  사용자가 선택한 문서 유형 힌트
        model:      사용할 모델 (None이면 config.OPENAI_MODEL)

    Returns:
        VLM이 반환한 JSON dict (실패 시 {"error": "..."} 반환)
    """
    _model = model or OPENAI_MODEL

    try:
        client = _get_openai_client()
    except EnvironmentError as e:
        return {"error": str(e)}

    # 이미지 base64 인코딩
    try:
        path = Path(image_path)
        ext = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        media_type = mime_map.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return {"error": f"이미지 파일을 찾을 수 없습니다: {image_path}"}
    except Exception as e:
        return {"error": f"이미지 로드 실패: {e}"}

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _build_user_prompt(type_hint)},
                    ],
                },
            ],
            max_tokens=1500,
            temperature=0,
        )
        raw_text = response.choices[0].message.content.strip()

    except Exception as e:
        return {"error": f"OpenAI API 호출 실패: {e}", "model_used": _model}

    parsed_json = _parse_json_from_text(raw_text)
    if parsed_json is None:
        return {
            "error": f"OpenAI 응답 JSON 파싱 실패: {raw_text[:300]}",
            "model_used": _model,
        }

    parsed_json["model_used"] = _model
    return parsed_json


# ══════════════════════════════════════════════
# 4. 프로바이더 라우터
# ══════════════════════════════════════════════

def call_vision(
    image_path: str,
    type_hint: str = "unknown",
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """
    VLM_PROVIDER 설정에 따라 Gemini 또는 OpenAI를 호출한다.

    Args:
        image_path: 로컬 이미지 경로
        type_hint:  문서 유형 힌트 ("receipt" | "tax_invoice" | ...)
        model:      사용할 모델 (None이면 각 프로바이더 기본값)
        provider:   "gemini" | "openai" (None이면 VLM_PROVIDER 환경변수 사용)

    Returns:
        VLM JSON dict (실패 시 {"error": "..."} 포함)
    """
    _provider = (provider or VLM_PROVIDER).lower()

    if _provider == "gemini":
        return call_vision_gemini(image_path, type_hint=type_hint, model=model)
    elif _provider == "openai":
        return call_vision_openai(image_path, type_hint=type_hint, model=model)
    else:
        return {"error": f"지원하지 않는 VLM_PROVIDER: '{_provider}'. 'gemini' 또는 'openai'를 사용하세요."}


# ══════════════════════════════════════════════
# 5. VLM 응답 → 표준 JSON 변환
#    (CLOVA parse_clova_response()와 동일한 출력 스키마)
# ══════════════════════════════════════════════

def parse_vision_response(
    image_path: str,
    type_hint: str = "unknown",
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """
    VLM을 호출하고 결과를 CLOVA 호환 표준 JSON으로 반환한다.

    이 함수의 반환 스키마는 clova_ocr_receipt.parse_clova_response()와 동일하다.
    → pipeline_service.py에서 call_clova_receipt + parse_clova_response를
      이 함수 하나로 대체할 수 있다.

    Args:
        image_path: 로컬 이미지 경로
        type_hint:  사용자가 업로드 시 선택한 문서 유형
        model:      사용할 모델 (None이면 각 프로바이더 기본값)
        provider:   "gemini" | "openai" (None이면 VLM_PROVIDER 환경변수 사용)

    Returns:
        표준 JSON dict + "validation" 키 (validate_result 결과 포함)
    """
    source_name = Path(image_path).name
    _provider = (provider or VLM_PROVIDER).lower()
    _model = model or (GEMINI_MODEL if _provider == "gemini" else OPENAI_MODEL)

    # ── VLM 호출 ────────────────────────────────
    vlm_raw = call_vision(image_path, type_hint=type_hint, model=model, provider=provider)

    # ── 오류 처리 ────────────────────────────────
    if "error" in vlm_raw:
        result = {
            "ocr_type":        "receipt",
            "source_file":     source_name,
            "ocr_engine":      f"vlm_{_provider}_{_model}",
            "infer_result":    "ERROR",
            "error":           vlm_raw["error"],
            "store":           {},
            "payment":         {},
            "items":           [],
            "total_amount":    None,
            "tax_amount":      None,
            "discount_amount": None,
            "model_used":      vlm_raw.get("model_used", _model),
            "provider":        _provider,
        }
        return validate_result(result)

    # ── 표준 스키마로 변환 ───────────────────────
    infer_result = vlm_raw.get("infer_result", "FAILED")

    result = {
        "ocr_type":     "receipt",
        "source_file":  source_name,
        "ocr_engine":   f"vlm_{_provider}_{_model}",
        "infer_result": infer_result,
        "provider":     _provider,

        # store 정보
        "store": {
            "name":     vlm_raw.get("store", {}).get("name"),
            "sub_name": None,
            "biz_num":  vlm_raw.get("store", {}).get("biz_num"),
            "address":  vlm_raw.get("store", {}).get("address"),
            "tel":      vlm_raw.get("store", {}).get("tel"),
        },

        # 결제 정보
        "payment": {
            "date":         vlm_raw.get("payment", {}).get("date"),
            "time":         vlm_raw.get("payment", {}).get("time"),
            "card_company": vlm_raw.get("payment", {}).get("card_company"),
            "card_number":  vlm_raw.get("payment", {}).get("card_number"),
            "confirm_num":  vlm_raw.get("payment", {}).get("confirm_num"),
        },

        # 품목 목록 (None 값 방어)
        "items": [
            {
                "name":       item.get("name"),
                "count":      item.get("count"),
                "unit_price": item.get("unit_price"),
                "amount":     item.get("amount"),
            }
            for item in (vlm_raw.get("items") or [])
        ],

        # 금액
        "total_amount":    vlm_raw.get("total_amount"),
        "tax_amount":      vlm_raw.get("tax_amount"),
        "discount_amount": vlm_raw.get("discount_amount"),

        # VLM 전용 메타
        "model_used": vlm_raw.get("model_used", _model),
        "confidence": vlm_raw.get("confidence"),
        "doc_type":   vlm_raw.get("doc_type"),
    }

    # FAILED인 경우 실패 사유 추가
    if infer_result == "FAILED":
        result["error"] = vlm_raw.get("fail_reason", "VLM 판독 실패")

    # ── validate_result 재사용 (CLOVA와 동일한 후처리 검증) ──
    return validate_result(result)


# ══════════════════════════════════════════════
# 6. DB 캐시 조회 헬퍼
#    (agent_logs에서 기존 VLM 결과 재사용)
# ══════════════════════════════════════════════

def get_cached_vision_result(conn, file_id: int) -> dict | None:
    """
    agent_logs에서 동일 file_id의 기존 VLM 결과를 조회한다.
    결과가 있으면 반환, 없으면 None 반환.

    재호출 방지 원칙:
        동일 file_id에 대해 VLM을 두 번 호출하지 않는다.
        → 캐시 결과가 있으면 VLM 호출 없이 바로 반환.

    Args:
        conn:    psycopg2 DB 커넥션
        file_id: 조회할 파일 ID

    Returns:
        캐시된 VLM 결과 dict 또는 None
    """
    import psycopg2.extras

    sql = """
        SELECT details
        FROM agent_logs
        WHERE agent_type_code = 'vision'
          AND status_code     = 'completed'
          AND (details->>'file_id')::int = %(file_id)s
        ORDER BY created_at DESC
        LIMIT 1
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"file_id": file_id})
        row = cur.fetchone()

    if row is None:
        return None

    details = row["details"]
    if isinstance(details, str):
        import json as _json
        details = _json.loads(details)
    return details


# ══════════════════════════════════════════════
# 7. 콘솔 요약 출력 (테스트용)
# ══════════════════════════════════════════════

def print_vision_summary(result: dict) -> None:
    """터미널에 VLM 파싱 결과 요약 출력"""
    sep = "─" * 54
    print(f"\n{sep}")
    print(f"  📄 파일:       {result.get('source_file', 'N/A')}")
    print(f"  🤖 엔진:       {result.get('ocr_engine', 'N/A')}")
    print(f"  🔍 인식 결과:  {result.get('infer_result', 'N/A')}  "
          f"(신뢰도: {result.get('confidence', 'N/A')})")
    print(f"  📑 문서 유형:  {result.get('doc_type', 'N/A')}")
    print(f"{sep}")

    if result.get("infer_result") in ("ERROR", "FAILED"):
        print(f"  ⚠️  오류: {result.get('error', '알 수 없는 오류')}")
        return

    store = result.get("store", {})
    pay   = result.get("payment", {})

    print(f"  🏪 업체명:     {store.get('name') or '─'}")
    print(f"  📋 사업자번호: {store.get('biz_num') or '─'}")
    print(f"  📅 결제일:     {pay.get('date') or '─'}")
    print(f"{sep}")

    items = result.get("items", [])
    if items:
        print(f"  🛒 품목 ({len(items)}개):")
        for item in items:
            name   = item.get("name") or "─"
            count  = item.get("count")
            amount = item.get("amount")
            count_str  = f"×{count}" if count else ""
            amount_str = f"{amount:,}원" if amount else "─"
            print(f"     · {name} {count_str}  →  {amount_str}")
    else:
        print(f"  🛒 품목:       인식된 항목 없음")

    total = result.get("total_amount")
    print(f"{sep}")
    print(f"  💰 총액:       {f'{total:,}원' if total else '─'}")

    val = result.get("validation", {})
    if val:
        print(f"{sep}")
        print(f"  ✅ 필수 필드:  {'완전' if val.get('has_required_fields') else '불완전 ⚠️'}")
        for w in val.get("warnings", []):
            print(f"  ⚠️  {w}")
        for e in val.get("errors", []):
            print(f"  ❌  {e}")

    print(f"{sep}\n")
