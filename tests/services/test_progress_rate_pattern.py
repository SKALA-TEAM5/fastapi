"""
누계공정률 추출 정규식 회귀 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
버그: 공식 서식의 라벨이 "누 계 공 정 율"처럼 글자 사이를 띄우고 률/율을 혼용해,
기존 정규식("누계공정률")이 매칭 실패 → header['공정률']=None → cumulative_progress_rate=0.
수정: 글자 간 공백과 률·율 변형을 허용.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ocr.parse_usage_statement import META_PATTERNS  # noqa: E402


def _extract_progress(text: str):
    for pat in META_PATTERNS["공정률"]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def test_spaced_label_with_yul():
    # 실제 부산 샘플 PDF의 페이지1 텍스트 형태
    text = "발 주 자 해운대구청 누 계 공 정 율 70%"
    assert _extract_progress(text) == "70"


def test_contiguous_label_with_ryul():
    assert _extract_progress("누계공정률: 55.5%") == "55.5"


def test_spaced_short_label():
    assert _extract_progress("공 정 율 70") == "70"


def test_decimal_value():
    assert _extract_progress("누 계 공 정 률 35.5 %") == "35.5"


def test_no_match_returns_none():
    assert _extract_progress("공사명 부산 해운대구 주상복합 신축공사") is None
