# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-06
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. _fake_result() : presenter 단위 테스트용 결과 객체 생성
# 2. ReasonTemplateTests : reason_code 선택 및 문구 템플릿 검증
# --------------------------------------------------------------------------
from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.agents.validator_agent.presenter import (
    _apply_reason_template,
    _classify_reason_code,
)


def _fake_item(name: str, *, allowed: bool, exception_summary: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        item=name,
        allowed=allowed,
        exception_summary=exception_summary,
    )


def _fake_result(
    *,
    status: str = "적절",
    total: float = 0,
    limit: float | None = None,
    exceeded: bool = False,
    progress_rate: float | None = None,
    required_usage_rate: float | None = None,
    required_used_amount: float | None = None,
    cumulative_used_amount: float | None = None,
    usage_shortfall_amount: float | None = None,
    rejection_reason: str = "",
    items: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        total=total,
        limit=limit,
        exceeded=exceeded,
        progress_rate=progress_rate,
        required_usage_rate=required_usage_rate,
        required_used_amount=required_used_amount,
        cumulative_used_amount=cumulative_used_amount,
        usage_shortfall_amount=usage_shortfall_amount,
        rejection_reason=rejection_reason,
        items=items or [],
    )


class ReasonTemplateTests(unittest.TestCase):
    def _build_reason(
        self,
        *,
        result: SimpleNamespace,
        law_ref: str = "",
        disallowed_items: list[str] | None = None,
        allowed_items: list[str] | None = None,
        exception_texts: list[str] | None = None,
    ) -> tuple[str, str]:
        disallowed = disallowed_items or []
        allowed = allowed_items or []
        exceptions = exception_texts or []
        reason_code = _classify_reason_code(
            result=result,
            disallowed_items=disallowed,
            allowed_items=allowed,
            exception_texts=exceptions,
        )
        reason = _apply_reason_template(
            reason_code=reason_code,
            result=result,
            law_ref=law_ref,
            law_basis=f"{law_ref} " if law_ref else "",
            disallowed_items=disallowed,
            allowed_items=allowed,
            exception_texts=exceptions,
        )
        self.assertFalse(reason.startswith(" "))
        self.assertEqual(reason.strip(), reason)
        self.assertNotIn("{items}", reason)
        self.assertNotIn("{law_ref}", reason)
        return reason_code, reason

    def test_rc_improper_scope_exclusion_01(self) -> None:
        result = _fake_result(status="부적절", items=[_fake_item("사무실용 소화기", allowed=False)])
        code, reason = self._build_reason(
            result=result,
            law_ref="「산안비 사용기준」 제7조제1항제2호에 따르면",
            disallowed_items=["사무실용 소화기"],
        )
        self.assertEqual(code, "improper_scope_exclusion")
        self.assertIn("사무실용 소화기", reason)
        self.assertIn("사용 범위를 벗어나", reason)
        self.assertIn("재제출", reason)
        self.assertNotIn("혼재", reason)
        self.assertNotIn("한도", reason)
        self.assertNotIn("공정률", reason)

    def test_rc_improper_scope_exclusion_02(self) -> None:
        result = _fake_result(status="부적절", items=[_fake_item("사무실용 소화기", allowed=False)])
        code, reason = self._build_reason(
            result=result,
            law_ref="",
            disallowed_items=["사무실용 소화기"],
        )
        self.assertEqual(code, "improper_scope_exclusion")
        self.assertTrue(reason.startswith("이번에 검토한 사무실용 소화기 항목은"))
        self.assertNotIn("에 따르면", reason)

    def test_rc_improper_limit_exceeded_01(self) -> None:
        result = _fake_result(status="부적절", total=3_500_000, limit=3_000_000, exceeded=True)
        code, reason = self._build_reason(result=result)
        self.assertEqual(code, "improper_limit_exceeded")
        self.assertIn("3,500,000원", reason)
        self.assertIn("3,000,000원", reason)
        self.assertIn("500,000원 초과", reason)
        self.assertIn("전환 처리", reason)
        self.assertNotIn("혼재", reason)

    def test_rc_improper_limit_exceeded_02(self) -> None:
        result = _fake_result(status="부적절", total=3_100_000, limit=3_000_000, exceeded=True)
        code, reason = self._build_reason(result=result)
        self.assertEqual(code, "improper_limit_exceeded")
        self.assertIn("100,000원 초과", reason)

    def test_rc_review_progress_shortfall_01(self) -> None:
        result = _fake_result(
            status="검토필요",
            progress_rate=42.5,
            required_usage_rate=0.5,
            required_used_amount=2_500_000,
            cumulative_used_amount=1_700_000,
            usage_shortfall_amount=800_000,
        )
        code, reason = self._build_reason(result=result)
        self.assertEqual(code, "review_progress_shortfall")
        self.assertIn("42.5%", reason)
        self.assertIn("800,000원 부족", reason)
        self.assertIn("집행 계획을 점검", reason)
        self.assertNotIn("부적절", reason)

    def test_rc_review_exception_or_conflict_01(self) -> None:
        result = _fake_result(
            status="검토필요",
            rejection_reason="허용 근거와 불가 근거가 충돌",
            items=[_fake_item("안전관리자 인건비", allowed=True, exception_summary="감리원은 제외한다")],
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["안전관리자 인건비"],
            exception_texts=["감리원은 제외한다"],
        )
        self.assertEqual(code, "review_exception_or_conflict")
        self.assertIn("안전관리자 인건비", reason)
        self.assertIn("감리원은 제외한다", reason)
        self.assertIn("소명 자료", reason)
        self.assertNotIn("('')", reason)

    def test_rc_review_exception_or_conflict_02_empty_exception_text(self) -> None:
        result = _fake_result(
            status="검토필요",
            rejection_reason="허용 근거와 불가 근거가 충돌",
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["안전관리자 인건비"],
            exception_texts=[],
        )
        self.assertEqual(code, "review_exception_or_conflict")
        self.assertIn("단서 조항", reason)
        self.assertNotIn("('')", reason)

    def test_rc_review_insufficient_basis_01(self) -> None:
        result = _fake_result(
            status="검토필요",
            rejection_reason="직접 근거 미확인",
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["스마트 안전모"],
        )
        self.assertEqual(code, "review_insufficient_basis")
        self.assertIn("스마트 안전모", reason)
        self.assertIn("허용 근거가 충분히 확인되지 않", reason)
        self.assertIn("소명 자료", reason)

    def test_rc_review_duplicate_cost_risk_01(self) -> None:
        result = _fake_result(
            status="검토필요",
            rejection_reason="공사비에 이미 계상된 항목",
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["안전망"],
        )
        self.assertEqual(code, "review_duplicate_cost_risk")
        self.assertIn("기포함된 항목", reason)
        self.assertIn("중복/이중 계상", reason)
        self.assertIn("추가 증빙", reason)

    def test_rc_improper_mixed_items_01(self) -> None:
        result = _fake_result(
            status="부적절",
            items=[
                _fake_item("용접작업용 소화기", allowed=True),
                _fake_item("사무실용 소화기", allowed=False),
            ],
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["용접작업용 소화기"],
            disallowed_items=["사무실용 소화기"],
        )
        self.assertEqual(code, "improper_mixed_items")
        self.assertIn("용접작업용 소화기(적정)", reason)
        self.assertIn("사무실용 소화기(부적정)", reason)
        self.assertIn("혼재", reason)
        self.assertIn("분리", reason)
        self.assertNotIn("사용 범위를 벗어나", reason)

    def test_rc_improper_mixed_items_02_limit_three_items(self) -> None:
        result = _fake_result(status="부적절")
        code, reason = self._build_reason(
            result=result,
            allowed_items=["A", "B", "C", "D"],
            disallowed_items=["X", "Y", "Z", "W"],
        )
        self.assertEqual(code, "improper_mixed_items")
        self.assertIn("A(적정), B(적정), C(적정)", reason)
        self.assertIn("X(부적정), Y(부적정), Z(부적정)", reason)
        self.assertNotIn("D(적정)", reason)
        self.assertNotIn("W(부적정)", reason)

    def test_rc_appropriate_compliant_01(self) -> None:
        result = _fake_result(status="적절")
        code, reason = self._build_reason(
            result=result,
            law_ref="「산안비 사용기준」 제7조제1항제3호에 따르면",
            allowed_items=["안전모", "안전화"],
        )
        self.assertEqual(code, "appropriate_compliant")
        self.assertIn("모든 법적 기준을 충족", reason)
        self.assertIn("투명하게 관리", reason)
        self.assertNotIn("부족", reason)
        self.assertNotIn("초과", reason)
        self.assertNotIn("제외", reason)

    def test_rc_appropriate_compliant_02_without_law_ref(self) -> None:
        result = _fake_result(status="적절")
        code, reason = self._build_reason(
            result=result,
            law_ref="",
            allowed_items=[],
        )
        self.assertEqual(code, "appropriate_compliant")
        self.assertIn("사용 범위, 집행 한도", reason)
        self.assertFalse(reason.startswith(" "))

    def test_rc_complex_01_limit_precedence_over_scope(self) -> None:
        result = _fake_result(status="부적절", total=3_500_000, limit=3_000_000, exceeded=True)
        code, reason = self._build_reason(
            result=result,
            disallowed_items=["사무실용 소화기"],
        )
        self.assertEqual(code, "improper_limit_exceeded")
        self.assertIn("사무실용 소화기", reason)
        self.assertIn("500,000원 초과", reason)

    def test_rc_complex_02_mixed_precedence_over_progress(self) -> None:
        result = _fake_result(
            status="부적절",
            progress_rate=72.5,
            required_usage_rate=0.7,
            required_used_amount=35_000_000,
            cumulative_used_amount=34_500_000,
            usage_shortfall_amount=500_000,
        )
        code, reason = self._build_reason(
            result=result,
            allowed_items=["용접소화기"],
            disallowed_items=["사무실소화기"],
        )
        self.assertEqual(code, "improper_mixed_items")
        self.assertIn("혼재", reason)
        self.assertNotIn("500,000원 부족", reason)


if __name__ == "__main__":
    unittest.main()
