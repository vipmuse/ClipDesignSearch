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
