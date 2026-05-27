"""
classifier_profiles (CAT_02, CAT_06) 를 seed_legal_rule_profiles.json 기준으로
legal_rag.legal_rule_profiles 테이블에 UPSERT한다.

스키마 변경 없이 데이터만 갱신 (UPDATE / INSERT).
"""
import json
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://safety_user:safety_password@localhost:5432/safety",
)
RULE_CONFIG_PATH = Path("scripts/seed_legal_rule_profiles.json")


def upsert_classifier_profiles(target_categories: list[str] | None = None) -> None:
    config = json.loads(RULE_CONFIG_PATH.read_text(encoding="utf-8"))
    classifier_profiles: dict = config.get("classifier_profiles", {})

    if target_categories:
        classifier_profiles = {k: v for k, v in classifier_profiles.items() if k in target_categories}

    conn = psycopg.connect(DATABASE_URL)
    upserted = 0
    with conn:
        with conn.cursor() as cur:
            for category_code, profile in classifier_profiles.items():
                for profile_key, values in profile.items():
                    profile_id = f"profile:classifier:{category_code}:{profile_key}"
                    cur.execute(
                        """
                        INSERT INTO legal_rag.legal_rule_profiles
                          (profile_id, profile_scope, category_code, profile_key, values_json, metadata)
                        VALUES (%s, 'category', %s, %s, %s, '{"original_scope": "classifier_profile"}'::jsonb)
                        ON CONFLICT (profile_scope, category_code, profile_key)
                        DO UPDATE SET
                          values_json = EXCLUDED.values_json,
                          metadata    = legal_rag.legal_rule_profiles.metadata
                                        || '{"original_scope": "classifier_profile"}'::jsonb
                        """,
                        (
                            profile_id,
                            category_code,
                            profile_key,
                            json.dumps(values, ensure_ascii=False),
                        ),
                    )
                    upserted += 1
                    print(f"  UPSERT {category_code} / {profile_key}")
    conn.close()
    print(f"\n완료: {upserted}개 UPSERT")


if __name__ == "__main__":
    targets = sys.argv[1:] or None  # 예: python update_classifier_profiles.py CAT_02 CAT_06
    print(f"대상 카테고리: {targets or '전체'}")
    upsert_classifier_profiles(target_categories=targets)
