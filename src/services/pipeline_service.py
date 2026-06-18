# --------------------------------------------------------------------------
# 작성자   : 이현수(kacalu0930)
# 작성일   : 2026-05-11
# 수정일   : 2026-06-18 (Clova→VLM 전환: 죽은 clova import·CLI 가드 제거)
#
# ※ 이 모듈은 독립 실행 CLI 도구로, 다른 모듈에서 import되지 않음(운영 API/Orchestrator 경로 아님).
#    → "삭제 후보 보고" 참고: 사용 여부 검토 필요.
#
# [ 주요 함수 정의 ]  (CLI 파이프라인 단계)
#
# 1. main()                    : CLI 진입점
# 2. step1_parse_usage()       : 1단계 — 사용내역서 파싱
# 3. step2_ocr_receipts()      : 2단계 — 영수증 OCR(VLM)
# 4. step2b_parse_tax_invoices(): 2b단계 — 세금계산서 파싱
# 5. step3_match()             : 3단계 — 매칭
# 6. print_pipeline_summary()  : 결과 요약 출력
# --------------------------------------------------------------------------
"""
산업안전관리비 AI 검증 시스템 — 전체 실행 서비스
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
네 가지 모듈을 순서대로 실행해 최종 매칭 결과를 출력한다.

  1. src/ocr/parse_usage_statement  : 사용내역서 PDF 파싱
  2. src/ocr/clova_ocr_receipt      : 영수증 이미지 배치 OCR
  2b. src/ocr/parse_tax_invoice     : 세금계산서 파싱 (PDF or 이미지 자동 분기)
  3. src/services/matching_service : 2-way 매칭 — 월 단위 날짜 비교 (임계값 0.85 / 0.75)

파이프라인 흐름:
  입력: 사용내역서 PDF + 영수증 이미지 폴더
        + (선택) 세금계산서 파일/폴더
        + (선택) 현장사진 텍스트 JSON
    ↓
  Step 1:  사용내역서 파싱
    ↓
  Step 2:  영수증 OCR 일괄 처리
    ↓
  Step 2b: 세금계산서 파싱 (PDF or 이미지 자동 분기)
    ↓
  Step 3:  2-way 매칭
    ↓
  출력: JSON 결과 파일 + 콘솔 요약 리포트

CLI 사용법:
    # 기본 (사용내역서 + 영수증)
    python -m src.services.pipeline_service \\
        --usage  사용내역서.pdf \\
        --receipts receipts/

    # 세금계산서 포함
    python -m src.services.pipeline_service \\
        --usage  사용내역서.pdf \\
        --receipts receipts/ \\
        --tax-invoices 세금계산서/ \\
        --photos photo_texts.json \\
        --output results/

설치:
    pip install pdfplumber requests Pillow psycopg2-binary python-dotenv
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════
# 모듈 임포트
# ══════════════════════════════════════════════

# src/services/ → src/ → 프로젝트 루트 순으로 올라가 sys.path에 추가
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.ocr.parse_usage_statement import (
    parse_pdf,
    print_summary as print_usage_summary,
)
# [Clova→VLM 리팩토링] 죽은 clova import(SUPPORTED_EXTS/CLOVA_SECRET/URL 등) 제거.
#   SUPPORTED_EXTS는 receipt_validator에서 가져오고, main()의 'OCR_ENGINE==clova' 가드도 삭제됨.
from src.ocr.receipt_validator import SUPPORTED_EXTS
from src.ocr import ocr_engine
from src.ocr.parse_tax_invoice import (
    parse_tax_invoice,
    process_folder as tax_invoice_process_folder,
    save_result as tax_invoice_save_result,
    print_summary as print_tax_invoice_summary,
    ALL_EXTS as TAX_INVOICE_EXTS,
)
from src.services.matching_service import (
    match_all_usage_to_receipts,
    save_match_result,
    print_batch_summary,
    THRESHOLD_MATCHED,
    THRESHOLD_REVIEW,
)


# ══════════════════════════════════════════════
# 파이프라인 단계별 함수
# ══════════════════════════════════════════════

def step1_parse_usage(pdf_path: str, output_dir: Path) -> dict:
    """
    Step 1: 사용내역서 PDF 파싱

    Args:
        pdf_path   : 사용내역서 PDF 경로
        output_dir : 중간 결과 저장 폴더

    Returns:
        parse_usage_statement.py 표준 JSON dict
    """
    print("\n" + "═" * 60)
    print("  [Step 1/3]  사용내역서 파싱")
    print("═" * 60)
    print(f"  입력: {pdf_path}")

    parsed = parse_pdf(pdf_path)
    print_usage_summary(parsed)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"step1_usage_parsed_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON 저장: {out_path}")

    return parsed


def step2_ocr_receipts(
    receipts_dir: str,
    output_dir: Path,
    secret: str = "",
    url: str = "",
) -> list[dict]:
    """
    Step 2: 영수증 이미지 폴더 배치 OCR

    Args:
        receipts_dir : 영수증 이미지 폴더 경로
        output_dir   : OCR JSON 저장 폴더
        secret       : CLOVA OCR Secret Key
        url          : CLOVA OCR 엔드포인트 URL

    Returns:
        OCR 결과 dict 리스트 (clova_ocr_receipt.py 표준 JSON)
    """
    print("\n" + "═" * 60)
    print("  [Step 2/3]  영수증 OCR 일괄 처리")
    print("═" * 60)

    folder = Path(receipts_dir)
    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])

    if not images:
        print(f"  [경고] 지원 이미지 없음: {folder}")
        return []

    ocr_out_dir = output_dir / "ocr_results"
    ocr_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  이미지 폴더: {folder}  ({len(images)}개)")
    print(f"  OCR 결과:   {ocr_out_dir}\n")

    results = []
    for i, img in enumerate(images, 1):
        print(f"  [{i}/{len(images)}]", end=" ")
        result = ocr_engine.parse_receipt(str(img))
        result["source_file"] = img.name
        results.append(result)
        if i < len(images):
            time.sleep(0.3)

    success = sum(1 for r in results if r.get("infer_result") == "SUCCESS")
    fail    = len(results) - success
    print(f"\n  ✅ OCR 완료: 총 {len(results)}개 | 성공 {success}개 | 실패 {fail}개")

    return results


def step2b_parse_tax_invoices(
    tax_invoices_path: str,
    output_dir: Path,
    secret: str,
    url: str,
) -> list[dict]:
    """
    Step 2b: 세금계산서 파싱 (PDF or 이미지 자동 분기)

    Args:
        tax_invoices_path : 세금계산서 파일 경로 (단일 파일 또는 폴더)
        output_dir        : 파싱 결과 JSON 저장 폴더
        secret            : CLOVA OCR Secret Key
        url               : CLOVA OCR 엔드포인트 URL

    Returns:
        세금계산서 파싱 결과 dict 리스트
    """
    print("\n" + "═" * 60)
    print("  [Step 2b/3]  세금계산서 파싱")
    print("═" * 60)

    tax_path = Path(tax_invoices_path)
    tax_out_dir = output_dir / "tax_invoice_results"
    tax_out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if tax_path.is_file():
        print(f"  입력 파일: {tax_path.name}")
        parsed = parse_tax_invoice(str(tax_path), secret=secret, url=url)
        print_tax_invoice_summary(parsed)

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = tax_out_dir / f"{tax_path.stem}_tax_invoice_{ts}.json"
        tax_invoice_save_result(parsed, str(out_path))
        print(f"  💾 저장: {out_path}")
        results.append(parsed)

    elif tax_path.is_dir():
        files = sorted([f for f in tax_path.iterdir()
                        if f.suffix.lower() in TAX_INVOICE_EXTS])
        if not files:
            print(f"  [경고] 세금계산서 파일 없음: {tax_path}")
            return []

        print(f"  입력 폴더: {tax_path.name}  ({len(files)}개 파일)")
        print(f"  결과 저장: {tax_out_dir}\n")

        for i, f in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] 처리 중: {f.name}")
            parsed = parse_tax_invoice(str(f), secret=secret, url=url)
            print_tax_invoice_summary(parsed)

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = tax_out_dir / f"{f.stem}_tax_invoice_{ts}.json"
            tax_invoice_save_result(parsed, str(out_path))
            print(f"  💾 저장: {out_path}")
            results.append(parsed)
            time.sleep(0.2)
    else:
        print(f"  [오류] 경로를 찾을 수 없습니다: {tax_path}")
        return []

    valid   = sum(1 for r in results if r.get("validation", {}).get("is_valid"))
    invalid = len(results) - valid
    print(f"\n  ✅ 세금계산서 파싱 완료: 총 {len(results)}개 | 유효 {valid}개 | 검토 필요 {invalid}개")

    return results


def step3_match(
    usage_statement: dict,
    receipts: list[dict],
    output_dir: Path,
    threshold_matched: float = THRESHOLD_MATCHED,
    threshold_review:  float = THRESHOLD_REVIEW,
    tax_invoices: list[dict] = None,
) -> dict:
    """
    Step 3: 2-way 매칭 (사용내역서 ↔ 영수증)

    Args:
        usage_statement    : Step 1 결과 (사용내역서 JSON)
        receipts           : Step 2 결과 (OCR JSON 리스트)
        output_dir         : 최종 결과 저장 폴더
        threshold_matched  : matched 임계값 (기본 0.85)
        threshold_review   : review_needed 하한 임계값 (기본 0.75)
        tax_invoices       : Step 2b 결과 (세금계산서 JSON 리스트, 선택)

    Returns:
        match_all_usage_to_receipts 결과 dict
    """
    tax_invoices = tax_invoices or []

    print("\n" + "═" * 60)
    print("  [Step 3/3]  2-way 매칭")
    print("═" * 60)
    items_count = len(usage_statement.get("line_items") or usage_statement.get("items", []))
    print(f"  사용내역 항목: {items_count}개")
    print(f"  영수증:        {len(receipts)}개")
    print(f"  세금계산서:    {len(tax_invoices)}개")
    print(f"  임계값:        matched≥{threshold_matched}  /  review≥{threshold_review}")
    print()

    batch = match_all_usage_to_receipts(
        usage_statement,
        receipts,
        threshold=threshold_review,
        threshold_matched=threshold_matched,
    )

    if tax_invoices:
        batch["tax_invoices_context"] = {
            "count":        len(tax_invoices),
            "valid_count":  sum(1 for t in tax_invoices if t.get("validation", {}).get("is_valid")),
            "total_amount": sum(t.get("total_amount") or 0 for t in tax_invoices),
            "invoices":     tax_invoices,
        }

    return batch


# ══════════════════════════════════════════════
# 콘솔 최종 요약 출력
# ══════════════════════════════════════════════

def print_pipeline_summary(
    usage: dict,
    receipts: list[dict],
    batch: dict,
    saved_path: str,
    elapsed: float,
    tax_invoices: list[dict] = None,
):
    """파이프라인 전체 실행 결과 최종 요약 출력"""
    tax_invoices = tax_invoices or []
    sep = "═" * 60
    s   = batch.get("summary", {})
    th  = batch.get("thresholds", {})

    print(f"\n{sep}")
    print("  🏁 파이프라인 완료")
    print(f"{sep}")
    print(f"  사용내역서:      {usage.get('source_file', '-')}")
    print(f"  처리된 영수증:   {len(receipts)}개  "
          f"(OCR 성공 {sum(1 for r in receipts if r.get('infer_result') == 'SUCCESS')}개)")
    if tax_invoices:
        valid_ti     = sum(1 for t in tax_invoices if t.get("validation", {}).get("is_valid"))
        total_ti_amt = sum(t.get("total_amount") or 0 for t in tax_invoices)
        print(f"  세금계산서:      {len(tax_invoices)}건  "
              f"(유효 {valid_ti}건, 합계 {total_ti_amt:,}원)")
    print(f"  매칭 항목:       {s.get('total', 0)}개")
    print(f"{sep}")

    total     = s.get("total", 0)
    matched   = s.get("matched", 0)
    review    = s.get("review_needed", 0)
    unmatched = s.get("unmatched", 0)
    rejected  = s.get("rejected", 0)
    pass_rate = round(matched / total * 100, 1) if total else 0.0

    thr_matched = th.get("matched", THRESHOLD_MATCHED)
    thr_review  = th.get("review",  THRESHOLD_REVIEW)
    print(f"  ✅ matched        {matched:>4}개  ({s.get('match_rate_pct', 0):.1f}%)  — 자동 통과 (≥{thr_matched})")
    print(f"  🔍 review_needed  {review:>4}개  ({s.get('review_rate_pct', 0):.1f}%)  — 담당자 검토 필요")
    print(f"  ❌ unmatched      {unmatched:>4}개  — 자동 반려 (<{thr_review})")
    print(f"  🚫 rejected       {rejected:>4}개  — 영수증 품목명 없음 등")
    print(f"{sep}")
    print(f"  통과율:          {pass_rate:.1f}%  (matched / total)")
    print(f"  처리 시간:       {elapsed:.1f}초")
    print(f"  결과 파일:       {saved_path}")
    print(f"{sep}\n")


# ══════════════════════════════════════════════
# CLI 진입점
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="산업안전관리비 AI 검증 시스템 — 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.services.pipeline_service --usage 사용내역서.pdf --receipts receipts/
  python -m src.services.pipeline_service --usage 사용내역서.pdf --receipts receipts/ --tax-invoices 세금계산서/
  python -m src.services.pipeline_service --usage 사용내역서.pdf --receipts receipts/ --photos photos.json --output results/
  python -m src.services.pipeline_service --usage 사용내역서.pdf --receipts receipts/ --threshold-matched 0.90 --threshold-review 0.80
        """,
    )

    parser.add_argument("--usage",   required=True, help="사용내역서 PDF 파일 경로")
    parser.add_argument("--receipts", required=True, help="영수증 이미지 폴더 경로")
    parser.add_argument("--tax-invoices", default=None, dest="tax_invoices",
                        help="세금계산서 파일 또는 폴더 경로 (선택)")
    parser.add_argument("--photos", default=None,
                        help='현장사진 텍스트 JSON 파일 경로')
    parser.add_argument("--output", default=None,
                        help="결과 저장 폴더 (기본: 사용내역서 폴더 내 pipeline_results/)")
    parser.add_argument("--secret", default=None,
                        help="(deprecated) 과거 CLOVA OCR Secret — 더 이상 사용하지 않음")
    parser.add_argument("--url", default=None,
                        help="(deprecated) 과거 CLOVA OCR URL — 더 이상 사용하지 않음")
    parser.add_argument("--threshold-matched", type=float, default=THRESHOLD_MATCHED,
                        help=f"matched 임계값 (기본: {THRESHOLD_MATCHED})")
    parser.add_argument("--threshold-review", type=float, default=THRESHOLD_REVIEW,
                        help=f"review_needed 하한 임계값 (기본: {THRESHOLD_REVIEW})")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="OCR를 건너뛰고 --receipts 폴더 내 기존 JSON 파일을 사용")
    parser.add_argument("--verbose", action="store_true",
                        help="개별 매칭 결과 상세 출력")

    args = parser.parse_args()
    start_time = time.time()

    usage_path    = Path(args.usage)
    receipts_path = Path(args.receipts)

    if not usage_path.exists():
        print(f"[오류] 사용내역서 파일을 찾을 수 없습니다: {usage_path}")
        sys.exit(1)
    if not receipts_path.is_dir():
        print(f"[오류] 영수증 폴더를 찾을 수 없습니다: {receipts_path}")
        sys.exit(1)

    output_dir = (
        Path(args.output) if args.output
        else usage_path.parent / "pipeline_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # OCR은 VLM 전면 사용. secret/url 인자는 더 이상 사용하지 않으며 하위호환용으로만 유지.
    secret = args.secret or ""
    url    = args.url    or ""

    print("\n" + "═" * 60)
    print("  산업안전관리비 AI 검증 시스템 — 통합 파이프라인")
    print(f"  실행일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)
    print(f"  사용내역서:   {usage_path.name}")
    print(f"  영수증 폴더:  {receipts_path.name}")
    if args.tax_invoices:
        print(f"  세금계산서:   {args.tax_invoices}")
    print(f"  결과 폴더:    {output_dir}")
    print(f"  임계값:       matched≥{args.threshold_matched}  /  review≥{args.threshold_review}")

    # Step 1
    usage_statement = step1_parse_usage(str(usage_path), output_dir)
    if usage_statement.get("parse_status") == "FAILED":
        print("\n[오류] 사용내역서 파싱 실패 — 파이프라인 중단")
        sys.exit(1)

    # Step 2
    if args.skip_ocr:
        print("\n" + "═" * 60)
        print("  [Step 2/3]  영수증 OCR — skip (기존 JSON 재사용)")
        print("═" * 60)
        json_files = [f for f in sorted(receipts_path.glob("*.json"))
                      if not f.name.endswith("_raw.json")]
        receipts = []
        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as f:
                    receipts.append(json.load(f))
                print(f"  로드: {jf.name}")
            except Exception as e:
                print(f"  [경고] 로드 실패 {jf.name}: {e}")
        print(f"\n  ✅ 기존 JSON {len(receipts)}개 로드 완료")
    else:
        receipts = step2_ocr_receipts(str(receipts_path), output_dir, secret, url)

    if not receipts:
        print("\n[경고] 처리된 영수증이 없습니다. 매칭을 건너뜁니다.")
        sys.exit(0)

    # Step 2b
    tax_invoices: list[dict] = []
    if args.tax_invoices:
        tax_invoices = step2b_parse_tax_invoices(args.tax_invoices, output_dir, secret, url)
    else:
        print("\n" + "═" * 60)
        print("  [Step 2b/3]  세금계산서 파싱 — skip (--tax-invoices 미지정)")
        print("═" * 60)

    # Step 3
    batch = step3_match(
        usage_statement, receipts, output_dir,
        threshold_matched=args.threshold_matched,
        threshold_review=args.threshold_review,
        tax_invoices=tax_invoices,
    )

    if args.verbose:
        from src.services.matching_service import print_match_result
        for r in batch["results"]:
            print_match_result(r)

    print_batch_summary(batch)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"match_result_{ts}.json"
    saved    = save_match_result(batch, str(out_path))

    elapsed = time.time() - start_time
    print_pipeline_summary(usage_statement, receipts, batch, saved, elapsed,
                           tax_invoices=tax_invoices)


if __name__ == "__main__":
    main()
