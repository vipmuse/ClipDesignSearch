"""이미지 업로드 → 유사 디자인 검색 웹 서버 (FastAPI).

  python webapp/server.py            # http://127.0.0.1:8000
전제: outputs/index 에 FAISS 인덱스가 있어야 함 (src/embed.py build 로 생성).
"""
import io
import os
import sys
import threading

import faiss
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from embed import encode_pil, encode_text, load_records, load_tuned  # noqa: E402

CONFIG = os.path.join(ROOT, "configs/lora_clip.yaml")
ADAPTER = os.path.join(ROOT, "outputs/lora-clip-design/final")
INDEX_DIR = os.path.join(ROOT, "outputs/index")

app = FastAPI(title="ClipDesignSearch")
_lock = threading.Lock()          # 단일 모델 → 추론 직렬화
STATE = {}


def load():
    import torch
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_tuned(cfg, ADAPTER)
    model.to(device)
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    meta = load_records(os.path.join(INDEX_DIR, "meta.jsonl"))

    # 이미지→명칭(zero-shot)용 라벨 뱅크: 원래대로 meta의 전체 물품명칭(object_title) 사용.
    import json as _json
    labels = sorted({m.get("text", "").strip() for m in meta if m.get("text", "").strip()})
    label_vecs = encode_text(model, proc, labels, device)   # [L, D] 정규화됨
    vectors = index.reconstruct_n(0, index.ntotal)          # 재정렬용 전체 이미지 벡터 캐시

    # 로카르노 분류 명칭 사전 (9판)
    loc_path = os.path.join(ROOT, "data/locarno_9.json")
    loc = _json.load(open(loc_path, encoding="utf-8")) if os.path.exists(loc_path) else {"subclasses": {}, "classes": {}}

    # 출원(patentID)별 전체 도면 맵 — 팝업용. 인덱스(8000)가 아니라 전체 pairs.jsonl 사용
    # → 인덱스에 없는 도면까지 그 출원의 모든 뷰를 보여줌.
    from collections import defaultdict
    all_records = load_records(os.path.join(ROOT, "data/pairs.jsonl"))
    design_map = defaultdict(list)
    for gi, r in enumerate(all_records):
        design_map[r.get("design_id", "")].append(gi)

    STATE.update(model=model, proc=proc, index=index, meta=meta,
                 size=cfg["model"]["image_size"], device=device,
                 labels=labels, label_vecs=label_vecs, vectors=vectors, locarno=loc,
                 all_records=all_records, design_map=design_map)
    print(f"loaded: {index.ntotal} vectors, {len(labels)} labels, "
          f"{len(loc.get('subclasses', {}))} locarno names, device={device}")


@app.on_event("startup")
def _startup():
    load()


@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8").read()


@app.get("/api/image/{idx}")
def image(idx: int):
    meta = STATE["meta"]
    if idx < 0 or idx >= len(meta):
        raise HTTPException(404)
    path = meta[idx]["image"]
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


def _patent_no(design_id):
    """'USD0624278-20100928' → 'USD0624278' (출원/등록번호 부분)."""
    return (design_id or "").split("-")[0]


def _pack_one(i, s):
    m = STATE["meta"][i]
    did = m.get("design_id", "")
    return {
        "idx": int(i), "score": round(float(s), 4),
        "title": m.get("text", ""), "viewpoint": m.get("viewpoint", ""),
        "locarno": m.get("locarno", ""), "design_id": did,
        "patent_no": _patent_no(did),
        "num_drawings": len(STATE["design_map"].get(did, [])),
        "image_url": f"/api/image/{int(i)}",
    }


def _dedup_pack(idxs, scores, topk):
    """점수 내림차순 (idxs,scores)에서 출원번호(design_id) 중복 제거 후 상위 topk."""
    seen, out = set(), []
    for i, s in zip(idxs.tolist(), scores.tolist()):
        if i < 0:
            continue
        did = STATE["meta"][int(i)].get("design_id", "")
        if did in seen:
            continue
        seen.add(did)
        out.append(_pack_one(int(i), s))
        if len(out) >= topk:
            break
    return out


def _fmt_locarno(code):
    """'1403' → '14-03'."""
    code = (code or "").strip()
    return f"{code[:2]}-{code[2:]}" if len(code) == 4 else code


def _locarno_names(code):
    """로카르노 코드 → {'en':..,'ko':..}. 과(subclass) 우선, 없으면 류(class), 그래도 없으면 미분류."""
    loc = STATE.get("locarno", {})
    dash = _fmt_locarno(code)
    ent = loc.get("subclasses", {}).get(dash) or loc.get("classes", {}).get(dash[:2])
    if isinstance(ent, dict):
        return {"en": ent.get("en", "") or dash, "ko": ent.get("ko", "")}
    if isinstance(ent, str) and ent:
        return {"en": ent, "ko": ""}
    return {"en": "Unclassified", "ko": "미분류 / 기타"}


def _locarno_facets_pool(qvec, topn=10, pool=200):
    """유사도 상위 pool개 도면(출원번호 중복제거 안 함)에서 로카르노를 '유사도 가중합'으로
    랭킹 → top-N. 화면 표시 건수와 무관하게, 쿼리에 가까운 도면일수록 크게 반영.
    qvec: 이미지검색이면 이미지 임베딩, 텍스트검색이면 텍스트 임베딩."""
    idx = STATE["index"]
    p = min(pool, idx.ntotal)
    scores, idxs = idx.search(qvec.astype("float32"), p)
    meta = STATE["meta"]
    agg = {}                                        # code -> [유사도합, 도면수]
    for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        code = meta[i].get("locarno", "")
        if not code:
            continue
        a = agg.setdefault(code, [0.0, 0])
        a[0] += max(0.0, float(s))                  # 가까울수록(유사도↑) 가중치 큼
        a[1] += 1
    total = sum(v[0] for v in agg.values()) or 1.0
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:topn]
    out = []
    for code, (wsum, cnt) in ranked:
        nm = _locarno_names(code)
        out.append({
            "code": _fmt_locarno(code), "name": nm["en"], "name_ko": nm["ko"],
            "count": cnt, "pct": round(100 * wsum / total, 1),
        })
    return out


def _predict_labels(img_vec, topn=5):
    """이미지 임베딩 → 라벨 뱅크와의 코사인 유사도로 예측 명칭(zero-shot)."""
    sims = (STATE["label_vecs"] @ img_vec[0])           # [L]
    top = np.argsort(-sims)[:topn]
    return [{"label": STATE["labels"][j], "score": round(float(sims[j]), 4)} for j in top]


@app.post("/api/search")
def search(image: UploadFile = File(...), topk: int = 12,
           category: str = Form(""), alpha: float = Form(1.0)):
    """이미지 검색. category+alpha 주면 '개념 검색'(쿼리 융합):
       query = normalize(alpha*이미지 + (1-alpha)*카테고리텍스트).
       alpha=1.0 → 순수 이미지 유사도. alpha↓ → 예측/지정 카테고리 쪽으로 재정렬."""
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")

    alpha = max(0.0, min(1.0, float(alpha)))
    with _lock:
        img_vec = encode_pil(STATE["model"], STATE["proc"], [img], STATE["size"], STATE["device"])
        predicted = _predict_labels(img_vec)            # 이미지 → 예측 물품 카테고리
        used_cat = category.strip() or (predicted[0]["label"] if predicted else "")

        if alpha < 1.0 and used_cat:
            # 개념 검색(image→text→image): ①카테고리 텍스트로 후보 풀 확정 → ②풀 내에서
            # 이미지 유사도로 재정렬. 최종점수 = alpha*이미지유사 + (1-alpha)*카테고리유사.
            tvec = encode_text(STATE["model"], STATE["proc"], [used_cat], STATE["device"])
            pool = min(max(int(topk) * 25, 200), STATE["index"].ntotal)
            t_scores, t_idxs = STATE["index"].search(tvec, pool)
            ids, tsc = t_idxs[0], t_scores[0]
            keep = ids >= 0
            ids, tsc = ids[keep], tsc[keep]
            isc = STATE["vectors"][ids] @ img_vec[0]    # 풀 항목의 이미지 유사도
            final = alpha * isc + (1.0 - alpha) * tsc
            order = np.argsort(-final)                   # 전체 정렬 (중복 제거는 아래에서)
            sel_ids, sel_sc = ids[order], final[order]
        else:
            used_cat = ""                               # 순수 이미지 검색
            pool = min(max(int(topk) * 12, 120), STATE["index"].ntotal)
            sel_sc, sel_ids = STATE["index"].search(img_vec.astype("float32"), pool)
            sel_ids, sel_sc = sel_ids[0], sel_sc[0]

        # 로카르노 분포: 항상 이미지↔이미지 유사도 상위 도면(중복제거 X)의 가중합 기준
        facets = _locarno_facets_pool(img_vec)

    results = _dedup_pack(sel_ids, sel_sc, int(topk))   # 출원번호 중복 제거(표시용)
    return JSONResponse({
        "count": len(results),
        "predicted_labels": predicted,                  # 이미지→텍스트 변환 결과
        "used_category": used_cat, "alpha": alpha,
        "results": results,
        "locarno_facets": facets,                       # 이미지 유사도 기반 로카르노 top10
    })


@app.post("/api/search_text")
def search_text(query: str = Form(...), topk: int = 12):
    query = query.strip()
    if not query:
        raise HTTPException(400, "검색어를 입력하세요")
    with _lock:
        vec = encode_text(STATE["model"], STATE["proc"], [query], STATE["device"])
        pool = min(max(int(topk) * 12, 120), STATE["index"].ntotal)
        scores, idxs = STATE["index"].search(vec, pool)
        facets = _locarno_facets_pool(vec)              # 텍스트↔이미지 유사도 가중합 기준
    results = _dedup_pack(idxs[0], scores[0], int(topk))   # 출원번호 중복 제거(표시용)
    return JSONResponse({
        "count": len(results),
        "query": query,
        "results": results,
        "locarno_facets": facets,                       # 유사도 기반 로카르노 top10
    })


@app.get("/api/design/{design_id}")
def design(design_id: str):
    """한 출원(patentID)의 모든 도면 (팝업용). 전체 pairs.jsonl 기준."""
    gids = STATE["design_map"].get(design_id, [])
    recs = STATE["all_records"]
    draws = [{
        "gid": g, "image_url": f"/api/drawing/{g}",
        "viewpoint": recs[g].get("viewpoint", ""), "title": recs[g].get("text", ""),
    } for g in gids]
    return JSONResponse({
        "design_id": design_id, "patent_no": _patent_no(design_id),
        "title": recs[gids[0]].get("text", "") if gids else "",
        "count": len(draws), "drawings": draws,
    })


@app.get("/api/drawing/{gid}")
def drawing(gid: int):
    """전체 도면 맵의 gid로 이미지 서빙 (팝업용)."""
    recs = STATE["all_records"]
    if gid < 0 or gid >= len(recs):
        raise HTTPException(404)
    path = recs[gid]["image"]
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
