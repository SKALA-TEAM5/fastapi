"""
AWS S3 파일 페치 유틸리티
━━━━━━━━━━━━━━━━━━━━━━━━
storage_key 를 받아 S3 에서 파일 바이트를 반환한다.
환경변수 미설정 시 HTTPException(503) 을 반환하므로
호출부는 별도 예외 처리가 필요 없다.
"""

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException, status

from src.core.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET,
)


def _get_client():
    """boto3 S3 클라이언트 생성 (환경변수 검증 포함)"""
    if not S3_BUCKET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="S3_BUCKET 환경변수가 설정되지 않았습니다.",
        )
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
    )


def fetch_file(storage_key: str) -> bytes:
    """
    S3 에서 파일을 가져와 bytes 로 반환한다.

    Args:
        storage_key: S3 오브젝트 키 (DB files.storage_key 값)

    Returns:
        파일 바이트

    Raises:
        HTTPException 422 — 키가 존재하지 않거나 접근 불가
        HTTPException 503 — AWS 자격증명 오류 / 네트워크 오류
    """
    client = _get_client()
    try:
        response = client.get_object(Bucket=S3_BUCKET, Key=storage_key)
        return response["Body"].read()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "AccessDenied", "403", "404"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"S3 파일에 접근할 수 없습니다 (key={storage_key}, code={code})",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"S3 오류: {e}",
        )
    except BotoCoreError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AWS 연결 오류: {e}",
        )
