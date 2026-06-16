"""
사용내역서 업로드 검증 (프로젝트 제약)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

프로젝트 생성 API에는 공사기간 유효성 검증이 있지만, 사용내역서 파싱 단계에는
동일한 방어 로직이 없어 비정상적인 월 슬롯(usage_statements 행)이 자동 생성되는
버그가 있었다. 이 모듈은 파싱 결과를 DB에 적재하기 "전에" 프로젝트 제약을 검증한다.

검증 항목
  1) 보고 연월이 프로젝트 공사기간(construction_start_date ~ construction_end_date)
     범위 내인지 (월 단위, 시작·종료 월 포함). 연월을 확정할 수 없으면 거부.
  2) 사용내역서 공사명이 프로젝트 공사명과 일치하는지 (공백 제거 후 완전 일치).

설계 메모
  - 이 모듈은 stdlib(datetime)만 의존한다. DB 드라이버·웹 프레임워크를 import 하지
    않으므로 순수 단위 테스트가 가능하다.
  - 검증 실패는 UsageStatementValidationError 로 표면화하며, 호출 측(파이프라인)은
    이 예외 발생 시 어떤 슬롯도 생성하지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


class UsageStatementValidationError(Exception):
    """사용내역서가 프로젝트 제약(공사기간·공사명)을 위반할 때 발생."""


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼 (순수 함수)
# ─────────────────────────────────────────────────────────────


def _safe_date(value: Any) -> date | None:
    """YYYY-MM-DD 문자열/date 를 date 로 변환. 실패 시 None."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _first_day_of_month(d: date) -> date:
    """주어진 날짜의 해당 월 1일."""
    return d.replace(day=1)


def _normalize_name(value: Any) -> str:
    """공사명 비교용 정규화: 모든 공백(스페이스·탭·개행) 제거."""
    if value is None:
        return ""
    return "".join(str(value).split())


# 공사명에서 변별력 없는 일반 접미사(공사 유형). 길이가 긴 것부터 제거한다.
_GENERIC_SUFFIXES = (
    "신축공사", "증축공사", "개축공사", "재축공사", "대수선공사", "리모델링공사",
    "보수공사", "보강공사", "해체공사", "철거공사", "리모델링",
    "신축", "증축", "개축", "공사",
)


def _construction_core(name: Any) -> str:
    """
    공사명에서 변별력 없는 일반 접미사(신축공사/증축공사/공사 등)를 제거한
    '핵심부'를 반환한다. 핵심부가 없으면(접미사만으로 구성) 빈 문자열.

    예) "서울 강남구 사무용 빌딩 신축공사" → "서울강남구사무용빌딩"
        "신축공사" → ""   (변별 정보 없음)
    """
    core = _normalize_name(name)
    for suffix in _GENERIC_SUFFIXES:  # 긴 접미사 우선
        if core.endswith(suffix):
            core = core[: -len(suffix)]
            break
    return core


# ─────────────────────────────────────────────────────────────
# 공개 함수
# ─────────────────────────────────────────────────────────────


def to_report_month(year: Any, month: Any) -> date | None:
    """
    사용자가 화면에서 선택한 연·월(예: 2026, 6)을 해당 월 1일 date 로 변환한다.
    값이 없거나 올바르지 않으면 None 을 반환한다.
    """
    if year is None or month is None or year == "" or month == "":
        return None
    try:
        return date(int(year), int(month), 1)
    except (ValueError, TypeError):
        return None


def resolve_report_month(parsed: dict[str, Any]) -> date | None:
    """
    파싱 결과에서 사용내역서의 보고월(해당 월 1일)을 도출한다.

    line_items 첫 항목의 사용일자(사용일자 / used_on)를 기준으로 한다.
    항목이 없거나 날짜를 파싱할 수 없으면 None 을 반환한다.
    (None 은 "연월을 확정할 수 없음"을 의미하며, 검증 단계에서 거부 사유가 된다.)
    """
    line_items = parsed.get("line_items") or parsed.get("items") or []
    if not line_items:
        return None
    first = line_items[0]
    first_date = _safe_date(first.get("사용일자") or first.get("used_on"))
    if first_date is None:
        return None
    return _first_day_of_month(first_date)


def validate_usage_against_project(
    *,
    report_month: date | None,
    project: dict[str, Any],
    parsed_usage: dict[str, Any],
    expected_report_month: date | None = None,
) -> None:
    """
    파싱된 사용내역서가 제약을 만족하는지 검증한다.

    Args:
        report_month          : resolve_report_month() 결과 (None 이면 연월 미확정)
        project               : {"project_name", "construction_start_date",
                                 "construction_end_date", ...}
        parsed_usage          : 파싱 결과 (header.공사명 사용)
        expected_report_month : 사용자가 화면에서 선택한 월(슬롯)의 1일.
                                None 이면 선택 월 일치 검증을 건너뛴다.

    검증 순서:
        0) 연월 확정 가능 여부
        1) 공사기간(construction_start_date ~ construction_end_date) 범위
        2) 공사명 일치

    참고: 화면에서 선택한 월(expected_report_month)과 사용내역서의 실제 월이 달라도,
    실제 월이 공사기간 내라면 차단하지 않는다. 슬롯은 사용내역서의 실제 월(report_month)로
    등록된다(insert_usage_statement가 파싱된 월을 사용). 즉 선택 월은 참고용이며, 문서의
    실제 월이 우선한다.

    Raises:
        UsageStatementValidationError : 위 중 하나라도 위반 시.
            이 경우 호출 측은 슬롯(usage_statement)을 생성하지 않아야 한다.
    """
    start = _safe_date(project.get("construction_start_date"))
    end = _safe_date(project.get("construction_end_date"))

    # 0) 연월 확정 가능 여부
    if report_month is None:
        raise UsageStatementValidationError(
            "사용내역서에서 사용 연월을 확인할 수 없어 업로드를 거부합니다."
        )

    # 1) 공사기간 범위 검증 (월 단위, 시작·종료 월 포함)
    period_text = (
        f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
        if start and end
        else "프로젝트 공사기간"
    )
    if start is not None and report_month < _first_day_of_month(start):
        raise UsageStatementValidationError(
            f"사용내역서 연월({report_month:%Y-%m})이 공사기간({period_text})을 벗어났습니다."
        )
    if end is not None and report_month > _first_day_of_month(end):
        raise UsageStatementValidationError(
            f"사용내역서 연월({report_month:%Y-%m})이 공사기간({period_text})을 벗어났습니다."
        )

    # 2) 공사명 일치 검증 (핵심어 비교)
    #    일반 접미사(신축공사/증축공사/공사 등)는 변별력이 없어 제거한 '핵심부'로 비교한다.
    #    PDF 표 레이아웃 탓에 공사명이 일부만 추출될 수 있으므로(예: 전체 "서울 강남구 사무용
    #    빌딩 신축공사" 중 "신축공사"만), 핵심부끼리 한쪽이 다른 쪽을 포함하면 일치로 본다.
    #    - 핵심부가 명백히 다르면(예: '대전제조시설' vs '서울강남구사무용빌딩') → 차단
    #    - 추출값이 접미사뿐이거나 비어 변별 정보가 없으면 → 검증 건너뜀(차단하지 않음)
    header = parsed_usage.get("header") or {}
    parsed_name = header.get("공사명")
    project_name = project.get("project_name")
    parsed_core = _construction_core(parsed_name)
    project_core = _construction_core(project_name)
    if parsed_core and project_core and parsed_core not in project_core and project_core not in parsed_core:
        raise UsageStatementValidationError(
            f"사용내역서 공사명('{parsed_name}')이 "
            f"프로젝트 공사명('{project_name}')과 일치하지 않습니다."
        )
