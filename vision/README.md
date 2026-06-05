# Vision AI Workspace

사진 입력을 받아 적정성 여부를 판단하고, 결과를 JSON으로 반환하는 Vision AI 전용 작업 공간입니다.

## 범위

이 디렉터리에서 다루는 것:

- Vision AI agent
- 이미지 전처리 및 후처리
- 적정성 판단 로직
- JSON 출력 스키마
- 샘플 이미지 및 실험 예제
- BentoML 기반 모델 서빙 설정
- uv 기반 Python 의존성 관리

현재 모델:

- `models/ppe-detector.pt`: 안전모, 안전화, 안전벨트 착용/미착용 bbox detector
- `models/safety-net-classifier.pt`: 안전망 설치 여부 classifier

Kubernetes 배포에서는 모델 파일을 이 디렉터리에 커밋하지 않습니다.
모델은 MinIO에 저장하고, `SKALA-TEAM5/deploy` 레포의 `team5-vision` Deployment initContainer가 Pod 시작 시 `/models`로 내려받습니다.

```text
safety-files/models/vision/ppe-detector.pt
safety-files/models/vision/safety-net-classifier.pt
```

FastAPI 저장소의 `vision/` 디렉터리는 Vision 코드와 Docker 이미지 빌드를 담당하고, Kubernetes manifest는 `SKALA-TEAM5/deploy` 저장소의 `k8s/vision`에서 관리합니다.

## 입력 / 출력

입력:

- 보호구 점검 사진 파일 또는 파일 주소
- 안전망 점검 사진 파일 또는 파일 주소

출력:

- 적정성 판단 결과 JSON
- PPE bbox 렌더링에 필요한 JSON 좌표
- 안전망 상태 배너가 그려진 결과 이미지 경로/URL

예시:

```json
{
  "overall_status": "needs_review",
  "is_appropriate": null,
  "message": "안전화 항목은 검토가 필요합니다.",
  "ppe_status": "needs_review",
  "safety_net_review": {
    "status": "unclear",
    "is_appropriate": null,
    "confidence": 0.54,
    "reason": "이 사진에서는 안전망 설치 여부를 판단하기 어렵습니다."
  }
}
```

운영 API에서는 사진 업로드 시 선택된 점검 항목에 따라 모델을 따로 실행합니다.

- 보호구 점검: `POST /detect/ppe/source`
- 안전망 점검: `POST /detect/safety-net/source`
- Orchestrator 일괄 검토: `POST /vision/review`

컨테이너 실행 기준으로 입력 이미지 디렉토리는 `/data/files`, 결과 이미지 디렉토리는 `/data/vision_results`입니다.
보호구는 결과 이미지를 저장하지 않고 원본 이미지와 JSON bbox를 프론트에서 조합해 표시합니다.
안전망 결과 이미지는 `/data/vision_results/annotated/safety-net` 아래에 저장됩니다.
백엔드는 DB에서 가져온 파일 위치를 `source_uri`로 전달합니다.

```json
{
  "source_id": "image_12345",
  "source_uri": "/data/files/raw_image_022__0009.jpg"
}
```

백엔드는 Vision API 응답 JSON을 로그 테이블에 저장합니다.
PPE 화면은 원본 이미지와 `detections[].bbox_xyxy`를 사용해 박스 표시를 켜고 끌 수 있고, 안전망 화면은 `annotated_image_url`을 그대로 표시할 수 있습니다.

Orchestrator는 여러 장의 presigned URL을 한 번에 `/vision/review`로 전달할 수 있습니다.

```json
{
  "project_id": 1,
  "usage_statement_id": 123,
  "photos": [
    {
      "file_id": 10,
      "original_filename": "현장사진.jpg",
      "storage_key": "projects/1/site-photo/a.jpg",
      "evidence_type_code": "site_photo",
      "mime_type": "image/jpeg",
      "size_bytes": 12345,
      "presigned_url": "http://minio/bucket/projects/1/site-photo/a.jpg?signature=..."
    }
  ]
}
```

`/vision/review`는 부적정, 검토 필요, 판단 불가를 모두 `status_code=success`, `result_code=hil`로 반환합니다.
이미지 다운로드 실패 또는 모델 오류 같은 시스템 오류만 `status_code=fail`, `result_code=fail`로 반환합니다.

오케스트레이터 연동 시 URL 기준은 두 가지를 분리해서 봅니다.

- `presigned_url`: Vision 컨테이너가 직접 열 수 있는 주소여야 합니다. Docker 내부에서는 `localhost:9000`이 아니라 `http://minio:9000` 같은 compose 서비스명을 사용합니다.
- `annotated_image_url`: 프론트가 직접 열 수 있는 결과 이미지 주소입니다. `VISION_PUBLIC_BASE_URL`을 설정하면 절대 URL로 반환합니다.

로컬 프론트에서 결과 이미지를 직접 확인할 때:

```env
VISION_PUBLIC_BASE_URL=http://localhost:8002
```

compose 내부에서 다른 백엔드 서비스가 Vision API를 호출할 때:

```env
VISION_AGENT_BASE_URL=http://vision:8002
VISION_AGENT_REVIEW_PATH=/vision/review
VISION_AGENT_TIMEOUT_SECONDS=60
S3_PUBLIC_ENDPOINT_URL=http://minio:9000
```

안전망 classifier는 현재 `installed`, `missing`만 학습되어 있으므로 confidence가 낮은 경우 코드에서 `unclear`로 처리합니다.

## 디렉토리 원칙

모든 구현 코드는 `src/` 아래에 둡니다.
루트에는 프로젝트 설정 파일, BentoML entrypoint(`service.py`), 공통 문서만 둡니다.

권장 구조:

```text
src/
  agents/
    vision_review_agent/
  vision/
  prompts/
  schemas/
  services/
  repositories/
  core/
examples/
tests/
docs/
service.py
bentofile.yaml
pyproject.toml
uv.lock
```

## 폴더 설명

- `src/agents/`: 에이전트별 실행 로직
- `src/vision/`: 이미지 전처리, crop, 후처리
- `src/prompts/`: 비전 모델 프롬프트 (있으면)
- `src/schemas/`: 입력/출력 JSON 스키마
- `src/services/`: 모델 호출, 검증, 후처리
- `src/repositories/`: DB/스토리지 접근
- `src/core/`: 공통 설정, 상수, 유틸
- `examples/`: 샘플 이미지와 기대 결과
- `tests/`: 자동화 테스트
- `docs/`: 설계 및 작업 문서

## 현재 `src/` 코드 설명

현재 구현 코드는 FastAPI/BentoML 서빙, 모델 추론, 결과 이미지 생성, 로컬 배치 실행으로 나뉩니다.

```text
src/
  __init__.py
  main.py
  agents/
    vision_review_agent/
      __init__.py
      main.py
  core/
    __init__.py
    config.py
  schemas/
    __init__.py
    vision.py
  services/
    __init__.py
    vision_detection_service.py
  vision/
    __init__.py
    annotation.py
    image_loader.py
```

### `src/main.py`

Vision API의 FastAPI entrypoint입니다.

- `/health`: 서버 상태와 모델 파일 존재 여부 확인
- `/labels`: PPE detector 클래스 코드 확인
- `/detect/ppe/source`: 파일 경로를 받아 PPE 모델만 실행
- `/detect/safety-net/source`: 파일 경로를 받아 안전망 모델만 실행
- `/detect/ppe`: multipart 이미지 업로드로 PPE 모델 실행
- `/detect/safety-net`: multipart 이미지 업로드로 안전망 모델 실행
- `/detect`, `/detect/source`: 보호구와 안전망을 같이 실행하는 호환용 API
- `/vision/review`: Orchestrator가 여러 presigned URL 사진을 넘기는 일괄 검토 API

API에서 받은 `source_uri` 또는 `presigned_url`을 이미지로 읽고, 추론 결과 JSON을 반환합니다.
PPE 응답은 원본 이미지에 직접 그릴 bbox 좌표를 포함하고, 안전망 응답은 상태 배너가 그려진 `annotated_image_url`을 포함합니다.

### `src/core/config.py`

환경 변수와 기본 설정을 모아둔 파일입니다.

- 모델 경로: `VISION_MODEL_PATH`, `SAFETY_NET_MODEL_PATH`
- 입력/출력 경로: `VISION_INPUT_DIR`, `VISION_OUTPUT_DIR`
- 프론트 접근용 Vision 기준 URL: `VISION_PUBLIC_BASE_URL`
- confidence threshold: `VISION_REVIEW_CONF`, `SAFETY_NET_REVIEW_CONF`
- PPE 클래스 매핑: 모델 class id를 `안전벨트`, `안전화`, `안전모` 라벨로 변환

현재 PPE 모델 클래스는 다음 기준으로 매핑합니다.

```text
0 -> 01 안전벨트 착용
1 -> 02 안전벨트 미착용
2 -> 05 안전화 착용
3 -> 07 안전모 착용
4 -> 08 안전모 미착용
```

### `src/schemas/vision.py`

API 응답과 내부 결과 구조를 정의하는 Pydantic schema 파일입니다.

- `Detection`: bbox 하나에 대한 탐지 결과
- `EquipmentReview`: 안전모/안전화/안전벨트별 요약 판정
- `SafetyNetReview`: 안전망 설치 여부 판정
- `SourceDetectionRequest`: 파일 경로 기반 요청 body
- `PpeDetectionResponse`: PPE API 응답
- `SafetyNetDetectionResponse`: 안전망 API 응답
- `DetectionResponse`: 보호구와 안전망을 같이 실행하는 호환용 응답
- `VisionReviewRequest`, `VisionReviewPhoto`: Orchestrator가 `/vision/review`로 보내는 요청
- `VisionReviewResponse`, `VisionReviewTodo`: Orchestrator에 반환하는 검토 결과와 보완 필요 항목

백엔드가 DB 로그 테이블에 저장할 JSON 구조는 이 schema를 기준으로 보면 됩니다.

### `src/services/vision_detection_service.py`

실제 모델 추론과 판정 로직이 들어있는 서비스 파일입니다.

- `load_model()`: PPE YOLO 모델 로딩
- `load_safety_net_model()`: 안전망 classifier 모델 로딩
- `detect_ppe()`: PPE 모델 실행 후 장비별 판정 생성
- `detect_safety_net()`: 안전망 모델 실행 후 설치/미설치/판단 어려움 판정 생성
- `_detect_ppe_boxes()`: YOLO bbox 결과를 `Detection` schema로 변환
- `_build_reviews()`: 탐지 결과를 안전모/안전화/안전벨트별 요약으로 변환
- `_classify_safety_net()`: 안전망 classifier 결과를 `installed`, `missing`, `unclear`로 변환

PPE에서 미착용 항목은 부적정으로 처리하고, 착용 항목 confidence가 낮으면 검토 필요로 처리합니다.

### `src/vision/image_loader.py`

이미지를 불러오는 유틸 파일입니다.

- bytes 입력을 RGB 이미지로 변환
- `/data/files/...` 같은 절대 경로 이미지 로딩
- 파일명만 들어온 경우 `VISION_INPUT_DIR` 아래에서 이미지 검색
- http/https URL 이미지 로딩

백엔드가 DB에서 가져온 파일 위치를 `source_uri`로 넘기면, 이 파일의 함수가 실제 이미지를 열어줍니다.

### `src/vision/annotation.py`

결과 이미지를 생성하는 후처리 파일입니다.

- PPE 결과 이미지에 bbox와 라벨 그리기
- 안전망 결과 이미지에 설치/미설치/판단 어려움 배너 그리기
- 한글 폰트 로딩
- 작은 이미지는 결과 이미지용 캔버스를 최소 폭으로 확대해서 라벨 화질 개선
- bbox 좌표를 결과 이미지 크기에 맞춰 스케일링

JSON의 bbox 좌표는 원본 이미지 기준이고, 결과 이미지에 그릴 때만 시각화용으로 좌표를 확대합니다.

### `src/agents/vision_review_agent/main.py`

로컬 배치 실행용 스크립트입니다.

- `volumes/files` 아래 이미지를 한 번에 처리
- 파일명 기준으로 `ppe` 또는 `safety-net` 모델 선택
- `ppe1.jpg`, `ppe2.jpg` 같은 파일은 PPE 모델 실행
- `safety-net1.jpg`, `safety-net2.jpg` 같은 파일은 안전망 모델 실행
- 결과 JSON은 `volumes/vision_results/ppe`, `volumes/vision_results/safety-net` 아래 저장
- 결과 이미지는 `volumes/vision_results/annotated/ppe`, `volumes/vision_results/annotated/safety-net` 아래 저장

로컬에서 전체 샘플을 다시 돌릴 때는 루트에서 다음 명령을 사용합니다.

```bash
make vision-review
```

### `__init__.py`

각 디렉토리를 Python package로 인식시키기 위한 파일입니다.
대부분 비어 있으며, 직접 실행 로직은 들어가지 않습니다.

## 파일 규칙

- 에이전트 실행 파일은 `main.py`로 통일합니다.
- Python 파일과 디렉토리는 소문자 + 스네이크 케이스를 사용합니다.
- 서비스 파일은 `*_service.py`
- 저장소 파일은 `*_repository.py`
- 테스트 파일은 `test_*.py`
- 예제 JSON은 `sample_*.json`, `expected_*.json`

예시:

```text
src/agents/vision_review_agent/
  __init__.py
  main.py
  config.py
```

## 작업 규칙

- 새 기능 디렉토리를 루트에 직접 만들지 않습니다.
- 새 에이전트는 `src/agents/<agent_name>/` 아래에 추가합니다.
- 이미지 처리 코드는 `src/vision/` 아래에 둡니다.
- 프롬프트 파일은 `src/prompts/` 또는 에이전트 내부에 둡니다.
- 샘플 데이터는 `examples/` 아래에 둡니다.
- 개인 실험 파일, 임시 파일, 로컬 산출물은 커밋하지 않습니다.

## 브랜치 규칙

- 기능 개발: `feature/<name>`
- 버그 수정: `fix/<name>`
- 긴급 수정: `hotfix/<name>`

PR 대상은 기본적으로 `main`입니다.

## 개발 환경

- Python: 3.11.9
- 패키지 관리: `uv`

설치:

```bash
uv python install 3.11.9
uv sync
```
