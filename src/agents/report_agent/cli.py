from __future__ import annotations

"""ReportContext JSON을 ReportDraft JSON 보고서로 생성하는 CLI입니다."""

import argparse
import json
from pathlib import Path

from .agent import ReportAgent
from .schemas import ReportContext, ReportDraft


EXAMPLES_REPORT_AGENT_DIR = Path(__file__).resolve().parents[3] / "examples" / "report_agent"
DEFAULT_REPORT_OUTPUT = EXAMPLES_REPORT_AGENT_DIR / "sample_report_llm_output.json"


def load_report_context(path: str | Path) -> ReportContext:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReportContext.model_validate(data)


def resolve_report_output_path(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        return DEFAULT_REPORT_OUTPUT

    output = Path(output_path)
    if output.is_absolute() or output.parent != Path("."):
        return output
    return EXAMPLES_REPORT_AGENT_DIR / output.name


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
    output_json_path: str | Path | None = None,
    *,
    use_default_llm: bool = True,
) -> Path:
    context = load_report_context(input_context_path)
    draft = ReportAgent(use_default_llm=use_default_llm).generate(context)
    return write_report_draft_json(draft, resolve_report_output_path(output_json_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReportDraft JSON from ReportContext JSON.")
    parser.add_argument("input_context_json", help="Path to ReportContext JSON.")
    parser.add_argument(
        "output_report_json",
        nargs="?",
        help=(
            "Path to write the generated ReportDraft JSON. "
            "If omitted, writes to examples/report_agent/sample_report_llm_output.json. "
            "A filename without a directory is also written under examples/report_agent/."
        ),
    )
    args = parser.parse_args()

    output = generate_report_json(
        args.input_context_json,
        args.output_report_json,
    )
    print(output)


if __name__ == "__main__":
    main()
