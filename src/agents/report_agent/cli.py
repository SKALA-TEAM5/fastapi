from __future__ import annotations

"""ReportContext JSON을 ReportDraft JSON 보고서로 생성하는 CLI입니다."""

import argparse
import json
from pathlib import Path

from .agent import ReportAgent
from .schemas import ReportContext, ReportDraft


def load_report_context(path: str | Path) -> ReportContext:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReportContext.model_validate(data)


def write_report_draft_json(draft: ReportDraft, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(draft.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def generate_report_json(
    input_context_path: str | Path,
    output_json_path: str | Path,
    *,
    use_default_llm: bool = True,
) -> Path:
    context = load_report_context(input_context_path)
    draft = ReportAgent(use_default_llm=use_default_llm).generate(context)
    return write_report_draft_json(draft, output_json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReportDraft JSON from ReportContext JSON.")
    parser.add_argument("input_context_json", help="Path to ReportContext JSON.")
    parser.add_argument("output_report_json", help="Path to write the generated ReportDraft JSON.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable optional LLM text refinement and generate deterministic JSON only.",
    )
    args = parser.parse_args()

    output = generate_report_json(
        args.input_context_json,
        args.output_report_json,
        use_default_llm=not args.no_llm,
    )
    print(output)


if __name__ == "__main__":
    main()
