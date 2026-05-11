"""
PostgreSQL 연결/세션 유틸
━━━━━━━━━━━━━━━━━━━━━━━━
psycopg2 기반 커넥션 컨텍스트 매니저를 제공한다.
service 스키마를 기본 search_path로 설정한다.

사용 예:
    from src.repositories.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
"""

from __future__ import annotations

import contextlib
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

from src.core.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA


def _make_conn() -> PgConnection:
    """새 psycopg2 커넥션을 생성하고 search_path를 설정한다."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {DB_SCHEMA}, public")
    return conn


@contextlib.contextmanager
def get_connection() -> Generator[PgConnection, None, None]:
    """
    psycopg2 커넥션 컨텍스트 매니저.
    정상 종료 시 commit, 예외 발생 시 rollback한다.

    사용 예:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            # with 블록 종료 시 자동 commit
    """
    conn = _make_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_connection() -> bool:
    """DB 연결 가능 여부를 확인한다. 연결 성공 시 True 반환."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"[DB] 연결 실패: {e}")
        return False
