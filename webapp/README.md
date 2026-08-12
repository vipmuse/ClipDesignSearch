# 디자인 유사도 검색 웹앱

이미지를 업로드하면 학습된 MetaCLIP 2 + LoRA 모델로 유사한 디자인 도면을 찾아준다.

## 구성
- `server.py` — FastAPI 백엔드 (모델 로드 + FAISS 검색 + 이미지 서빙)
- `index.html` — 업로드 UI (드래그&드롭/붙여넣기, 결과 그리드)

## 사전 준비: FAISS 인덱스 빌드
```powershell
# 학습한 어댑터로 도면 DB 인덱싱 (데모는 --limit 로 일부만)
.\.venv\Scripts\python.exe src\embed.py build `
  --adapter outputs\lora-clip-design\final `
  --data data\pairs.jsonl --image-root . --index outputs\index --limit 8000
```

## 실행
```powershell
.\.venv\Scripts\python.exe webapp\server.py
# 브라우저에서 http://127.0.0.1:8000
```

## API
- `POST /api/search?topk=12` (multipart `image`) → 유사 도면 top-K JSON
- `GET  /api/image/{idx}` → 결과 도면 PNG

## 참고
- 인덱스를 늘리려면 `--limit`를 키우거나 제거(전체). 연도 데이터를 추가하려면
  `scripts\download_deeppatent2.py --years ...` → `scripts\build_pairs.py` → 재인덱싱.
- 현재 검색은 이미지→이미지(도면 유사도). 텍스트→이미지 검색은 `embed.py search` 참고.

## 학습 방법 선택 · 비교 (2026-08-13)

서버가 `outputs/methods/<이름>/index`의 방법별 인덱스를 전부 적재한다.

- `GET /api/methods` - 서빙 가능한 방법 목록 (+비활성 사유)
- `/api/search`, `/api/search_text`에 `method=<이름>` 또는 `methods=a,b`(나란히 비교, 최대 4개)
- 같은 model_id의 방법들은 백본 하나를 공유하고 named adapter로 전환한다.
  hires378만 별도 백본(378px). base는 어댑터를 끈 원본 백본.
- 인덱스 지문(check_index_meta)이 안 맞으면 그 방법만 비활성된다 - 시작 로그를 볼 것.
- 이미지→명칭 라벨 뱅크는 방법별 텍스트 인코더로 지연 인코딩된다 (방법당 첫 이미지
  검색이 몇 초 느림).

인덱스 준비:
```powershell
python scriptsun_ablation.py --arms baseline hobit tic loracap bigbatch hires378 --epochs 3
python src\embed.py build --config outputs\methodsaseline\config.resolved.yaml --adapter none --index outputs\methodsase\index --data data\subset_100k.jsonl
```
