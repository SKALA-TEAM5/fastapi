"""
챗봇 CLI 테스트 스크립트.

실행 방법:
    cd fastapi
    source .venv/bin/activate
    python scripts/chat.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.chatbot_service import stream_chat


async def main():
    print("=" * 50)
    print("산안비 챗봇 CLI (종료: Ctrl+C 또는 'q' 입력)")
    print("=" * 50)

    session_id = None

    while True:
        try:
            question = input("\n질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not question or question.lower() == "q":
            break

        print()
        async for event in stream_chat(question=question, session_id=session_id):
            if event.startswith("data: [DONE]"):
                print()
                break

            if not event.startswith("data: "):
                continue

            import json
            try:
                data = json.loads(event[6:])
            except Exception:
                continue

            t = data.get("type")
            v = data.get("value")

            if t == "session_id":
                session_id = v
            elif t == "status":
                print(f"[{v}]", end="\r")
            elif t == "intent":
                print(f"[의도: {v}]")
            elif t == "token":
                print(v, end="", flush=True)
            elif t == "sources":
                print(f"\n\n[출처] {', '.join(v)}")
            elif t == "error":
                print(f"\n[오류] {v}")


if __name__ == "__main__":
    asyncio.run(main())
