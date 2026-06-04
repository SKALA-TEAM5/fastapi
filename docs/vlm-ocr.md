# VLM OCR (영수증·증빙서류 파싱)

VLM OCR 파트는 영수증, 거래명세표, 세금계산서 등 증빙서류 이미지를 파싱하고, 사용내역서 항목과 2-way 매칭을 수행하는 파트입니다.

## 역할

- 증빙서류 이미지 → 구조화된 JSON 변환 (OCR 엔진 디스패치)
- 사용내역서 항목 ↔ 영수증 금액·날짜·업체명 2-way 매칭 (Hard Gate)
- 타임라인 검증 (공사 기간 외 집행, 공휴일/야간 결제 탐지)
- 매칭 결과 DB 저장 및 agent_logs 기록

## 관련 파일

- OCR 엔진 디스패처: `src/ocr/ocr_engine.py`
- VLM 호출 (Gemini/OpenAI): `src/ocr/vlm_ocr.py`
- 파싱 결과 검증: `src/ocr/receipt_validator.py`
- CLOVA OCR: `src/ocr/clova_ocr_receipt.py`
- 사용내역서 파싱: `src/ocr/parse_usage_statement.py`
- 세금계산서 파싱: `src/ocr/parse_tax_invoice.py`
- 매칭 엔진: `src/services/matching_service_monthly.py`
- 파이프라인 서비스: `src/services/usage_statement_pipeline_service.py`
- MinIO 클라이언트: `src/services/minio_client.py`
- OCR 라우터: `src/api/routers/parse.py`
- 매칭 라우터: `src/api/routers/matching.py`

## API 엔드포인트

| Method | Path | 역할 |
|---|---|---|
| `POST` | `/api/v1/ocr/parse` | 단일 증빙서류 OCR 파싱 |
| `POST` | `/api/v1/matching/run` | 사용내역서 ↔ 영수증 2-way 매칭 실행 |

### POST /api/v1/ocr/parse

DB `files` 테이블 레코드를 JSON으로 전달하면 `storage_key`로 MinIO에서 파일을 가져와 파싱합니다.

**문서 유형별 처리**

| uploaded_evidence_type_code | 파서 |
|---|---|
| `usage_statement` | pdfplumber |
| `receipt` | OCR 엔진 (VLM 또는 CLOVA, `OCR_ENGINE` 환경변수로 전환) |
| `transaction_statement` | pdfplumber (PDF) / OCR 엔진 (이미지) |
| `wage_statement` | 거래명세표 파서 공유 |
| `tax_invoice` | pdfplumber (PDF) / CLOVA (이미지) |

**Request Body 예시**

```json
{
  "id": 1,
  "project_id": 10,
  "uploaded_evidence_type_code": "receipt",
  "original_filename": "안전화_영수증_20260422.jpg",
  "storage_key": "projects/10/receipts/안전화_영수증_20260422.jpg",
  "mime_type": "image/jpeg",
  "size_bytes": 204800
}
```

**Response 예시 (OCR_ENGINE=vlm)**

```json
{
  "success": true,
  "data": {
    "ocr_result": {
      "receipt_id": "rec_a1b2c3d4",
      "doc_type": "receipt",
      "ocr_engine": "vlm",
      "infer_result": "SUCCESS",
      "vendor": "안전용품마트",
      "date": "2026-04-22",
      "total_amount": 150000,
      "items": [
        { "item_name": "안전화", "count": 3, "unit_price": 50000, "amount": 150000 }
      ],
      "validation": {
        "is_valid": true,
        "items_sum_match": true,
        "warnings": []
      }
    }
  }
}
```

## OCR 엔진 구조

```
ocr_engine.py          ← 디스패처 (OCR_ENGINE 환경변수로 분기)
    ├── vlm_ocr.py     ← VLM 호출 (Gemini / OpenAI)
    └── clova_ocr_receipt.py ← CLOVA OCR 호출
         ↓
receipt_validator.py   ← 엔진 무관 공통 검증 (금액 합산, 사업자번호 등)
```

### OCR 엔진 전환 방법

코드 수정 없이 `.env` 한 줄로 전환합니다.

```env
OCR_ENGINE=vlm    # Gemini / OpenAI VLM (수기 포함, 기본값)
OCR_ENGINE=clova  # NAVER CLOVA OCR (인쇄 영수증, 저비용)
```

### VLM 프로바이더 전환

```env
VLM_PROVIDER=gemini   # Google Gemini (기본값)
VLM_PROVIDER=openai   # OpenAI GPT-4o
```

## 매칭 엔진 (Hard Gate)

`POST /api/v1/matching/run`은 사용내역서 항목과 영수증 OCR 결과를 2-way 매칭합니다.

### 매칭 조건

| Gate | 조건 |
|---|---|
| 날짜 Gate | 같은 연월 기준 (월 경계 ±2일 허용) |
| 금액 Gate | 오차 1% 이내 |
| 업체명 Gate | 정규화 후 완전일치 (미기재 시 면제) |

### 판정 결과

| 결과 | 기준 |
|---|---|
| `matched` | 유사도 ≥ 0.85 |
| `review_needed` | 유사도 0.75 ~ 0.84 (Human-in-the-Loop 대상) |
| `unmatched` | 유사도 < 0.75 또는 Gate 미통과 |

### date_status 필드

| 값 | 의미 |
|---|---|
| `recognized` | 날짜 정상 인식 |
| `not_written` | 영수증에 날짜 미기재 |
| `unreadable` | 인식 불가 |

날짜 미기재 영수증은 `unmatched` 대신 `review_needed + review_flags`로 처리하여 사용자에게 재업로드를 안내합니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OCR_ENGINE` | `vlm` | OCR 엔진 선택 (`vlm` \| `clova`) |
| `VLM_PROVIDER` | `gemini` | VLM 프로바이더 (`gemini` \| `openai`) |
| `GEMINI_API_KEY` | — | Google Gemini API 키 (필수) |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | 기본 Gemini 모델 |
| `GEMINI_MODEL_FALLBACK` | `gemini-2.5-flash` | 폴백 모델 |
| `OPENAI_API_KEY` | — | OpenAI API 키 (VLM_PROVIDER=openai 시) |
| `OPENAI_MODEL` | `gpt-4o-mini` | 기본 OpenAI 모델 |
| `CLOVA_OCR_URL` | — | CLOVA OCR 엔드포인트 (OCR_ENGINE=clova 시) |
| `CLOVA_OCR_SECRET` | — | CLOVA OCR 시크릿 키 (OCR_ENGINE=clova 시) |
| `THRESHOLD_MATCHED` | `0.85` | matched 판정 임계값 |
| `THRESHOLD_REVIEW` | `0.75` | review_needed 판정 임계값 |

## 실행 환경

- Python `3.11.9`
- 패키지 관리: `uv`
- 외부 서비스: Google Gemini API (또는 OpenAI API), MinIO, PostgreSQL

## 처리 대상 문서

| 문서 종류 | 처리 방식 | 비고 |
|---|---|---|
| 사용내역서 (PDF) | pdfplumber | 수기 미지원, 프론트 수정 가능 |
| 카드영수증·간이영수증 | VLM 또는 CLOVA | `OCR_ENGINE` 환경변수로 전환 |
| 수기 영수증 | VLM 전용 | CLOVA 인식 불가 |
| 거래명세표 | VLM 또는 CLOVA | 품목명·단가·수량 추출 |
| 세금계산서 | pdfplumber / CLOVA | PDF 우선, 이미지는 CLOVA |
