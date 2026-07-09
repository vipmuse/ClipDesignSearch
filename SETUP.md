# 환경 셋업 기록 (검증 완료)

## 하드웨어 (LoRA 학습 적합성: 최상급 ✅)
| 항목 | 값 | 판정 |
|------|-----|------|
| GPU | NVIDIA RTX 5090 | Blackwell, sm_120 |
| VRAM | 31.8 GB | huge 모델 LoRA도 여유 |
| bf16 | 지원 + 실연산 스모크 테스트 통과 | ✅ |
| 시스템 RAM | 31 GB | 충분 |
| 작업 드라이브 | D: (SAMSUNG NVMe SSD) | 데이터 로딩 병목 없음 |

## 소프트웨어 (venv로 격리 — 전역 CPU torch 미변경)
- 위치: `D:\Workspace\ClipDesignSearch\.venv` (Python 3.13.5)
- **torch 2.11.0+cu128** ← Blackwell 필수 (기존 전역 `2.7.1+cpu`는 GPU 불가라 교체)
- transformers 5.13.0 / peft 0.19.1 / accelerate 1.14.0 / faiss-cpu 1.14.3

## 사용법
```powershell
# venv 활성화
.\.venv\Scripts\Activate.ps1

# 환경 재점검
python scripts\check_env.py

# 베이스 모델 다운로드 (models\ 로컬 저장)
python scripts\download_model.py

# 학습 / 인덱스 / 검색
python src\train.py --config configs\lora_clip.yaml
python src\embed.py build  --adapter outputs\lora-clip-design\final --data data\pairs.jsonl
python src\embed.py search --adapter outputs\lora-clip-design\final --text "무선 이어폰 케이스"
```

## 재현 설치 (다른 PC)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # torch 줄은 위에서 이미 설치됨
```

## 베이스 모델 선택 가이드 (32GB VRAM 기준)
- 기본: `facebook/metaclip-2-worldwide-huge-quickgelu` (여유롭게 학습 가능)
- 고해상도 정확도↑: `-huge-378` (배치 축소 필요)
- 빠른 PoC: `-s16` → 검증 후 huge로 확장
