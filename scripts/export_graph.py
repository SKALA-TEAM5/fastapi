"""
챗봇 LangGraph 그래프를 PNG로 저장하는 스크립트.

실행 방법:
    cd fastapi
    source .venv/bin/activate
    python scripts/export_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.chatbot_agent.agent import get_compiled_graph

output_path = Path(__file__).parent.parent / "docs" / "chatbot_graph.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

graph = get_compiled_graph()
png = graph.get_graph().draw_mermaid_png()
output_path.write_bytes(png)

print(f"저장 완료: {output_path}")
