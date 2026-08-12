"""이미지 업로드 → 유사 디자인 검색 웹 서버 (FastAPI). 학습 방법 선택·비교 지원.

  python webapp/server.py            # http://127.0.0.1:8000

전제: outputs/methods/<이름>/index 에 방법별 FAISS 인덱스가 있어야 함
(scripts/run_ablation.py 가 빌드, base는 src/embed.py build --adapter none).

방법별 적재 구조 (스펙 4장):
- model_id가 같은 방법들은 백본 하나를 공유하고 PeftModel named adapter로
  set_adapter 전환한다 (merge=False). hires378만 별도 백본.
- base(튜닝 전)는 disable_adapter 컨텍스트로 같은 백본을 쓴다.
- 인덱스마다 check_index_meta(..., n_vectors=ntotal)로 지문을 검증하고,
  불일치면 그 방법만 비활성한다 - 서버 전체를 죽이지 않는다 (스펙 7.1).
- meta는 내용 해시가 같으면 한 벌만 공유한다. 벡터 캐시(reconstruct_n 전량
  복사)는 제거하고 필요한 행만 reconstruct_batch로 꺼낸다 (스펙 4.2 RAM 대책).
"""
import hashlib
import io
import os
import sys
import threading
from contextlib import nullcontext

import faiss
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from embed import (check_index_meta, encode_pil, encode_text,   # noqa: E402
                   load_records)

METHODS_DIR = os.path.join(ROOT, "outputs/methods")
# 표시 순서 = 권장 순서 (부분집합 비교 결과: loracap I2I 승자, hires378 유일한 양쪽 개선)
METHOD_ORDER = ["loracap", "hires378", "baseline", "hobit", "tic", "bigbatch", "base"]

app = FastAPI(title="ClipDesignSearch")
_lock = threading.Lock()          # GPU 추론 + set_adapter 전환 직렬화
STATE = {}                        # 공유 자원 (locarno, 전체 도면 맵, meta 등)
METHODS = {}                      # name -> {index, meta, model_id, size, adapter_name, desc, ntotal}
DISABLED = {}                     # name -> 비활성 사유 (지문 불일치 등)
BACKBONES = {}                    # model_id -> {model, proc, is_peft}
LABEL_CACHE = {}                  # method -> label_vecs (지연 인코딩)


def _method_paths(name):
    """(resolved config 경로, 어댑터 경로 또는 'none'). base는 baseline 설정을 빌린다."""
    if name == "base":
        return os.path.join(METHODS_DIR, "baseline", "config.resolved.yaml"), "none"
    d = os.path.join(METHODS_DIR, name)
    return os.path.join(d, "config.resolved.yaml"), os.path.join(d, "final")


def _load_method(name, meta_pool):
    """방법 하나를 적재. 실패/불일치는 DISABLED에 사유를 남기고 None."""
    idx_dir = os.path.join(METHODS_DIR, name, "index")
    cfg_path, adapter = _method_paths(name)
    if not os.path.exists(os.path.join(idx_dir, "index_meta.json")):
        DISABLED[name] = "인덱스 없음"
        return None
    if not os.path.exists(cfg_path):
        DISABLED[name] = "config.resolved.yaml 없음"
        return None
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    if adapter != "none" and not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        DISABLED[name] = "어댑터 없음"
        return None

    index = faiss.read_index(os.path.join(idx_dir, "faiss.index"))
    ok, why = check_index_meta(idx_dir, cfg, adapter, cfg["train"]["data_path"],
                               n_vectors=index.ntotal)
    if not ok:
        DISABLED[name] = f"지문 불일치: {why}"
        return None

    # meta 공유: 내용 해시가 같으면 이미 읽은 리스트를 재사용 (방법 7개 × 수십 MB 절약)
    meta_path = os.path.join(idx_dir, "meta.jsonl")
    h = hashlib.sha1(open(meta_path, "rb").read()).hexdigest()
    if h not in meta_pool:
        meta_pool[h] = load_records(meta_path)
    desc = (cfg.get("method") or {}).get("description", "") if name != "base" \
        else "튜닝 전 베이스 모델 (비교 기준점)"
    return {"index": index, "meta": meta_pool[h], "meta_hash": h,
            "model_id": cfg["model"]["model_id"], "size": cfg["model"]["image_size"],
            "adapter": adapter, "adapter_name": None if adapter == "none" else name,
            "desc": desc, "ntotal": index.ntotal}


def _load_backbones(device):
    """model_id별로 백본 하나 + named adapter들. base는 어댑터 비활성 컨텍스트로 공유."""
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    by_model = {}
    for name, m in METHODS.items():
        by_model.setdefault(m["model_id"], []).append(name)
    for model_id, names in by_model.items():
        base = AutoModel.from_pretrained(model_id, attn_implementation="sdpa")
        proc = AutoProcessor.from_pretrained(model_id)
        peft = None
        for name in names:
            ad = METHODS[name]["adapter"]
            if ad == "none":
                continue
            if peft is None:
                peft = PeftModel.from_pretrained(base, ad, adapter_name=name)
            else:
                peft.load_adapter(ad, adapter_name=name)
        model = peft if peft is not None else base
        model.eval().to(device)
        BACKBONES[model_id] = {"model": model, "proc": proc, "is_peft": peft is not None}
        print(f"backbone {os.path.basename(model_id)}: adapters={[n for n in names if METHODS[n]['adapter'] != 'none']}")


def _use(name):
    """방법의 (모델, proc, size, 어댑터 컨텍스트). _lock 안에서 쓸 것."""
    m = METHODS[name]
    bb = BACKBONES[m["model_id"]]
    ctx = nullcontext()
    if bb["is_peft"]:
        if m["adapter_name"]:
            bb["model"].set_adapter(m["adapter_name"])
        else:
            ctx = bb["model"].disable_adapter()   # base: 어댑터 전부 끄고 원본 백본
    return bb["model"], bb["proc"], m["size"], ctx


def load():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_pool = {}
    for name in METHOD_ORDER:
        m = _load_method(name, meta_pool)
        if m is not None:
            METHODS[name] = m
    if not METHODS:
        raise RuntimeError(f"서빙 가능한 방법이 없습니다. 비활성 사유: {DISABLED}")

    # /api/image 는 방법과 무관해야 한다 → 모든 활성 방법의 meta가 같아야 한다.
    hashes = {m["meta_hash"] for m in METHODS.values()}
    if len(hashes) > 1:
        # 다른 데이터로 빌드된 인덱스가 섞여 있다 - 가장 많은 해시만 남긴다
        from collections import Counter
        keep = Counter(m["meta_hash"] for m in METHODS.values()).most_common(1)[0][0]
        for name in [n for n, m in METHODS.items() if m["meta_hash"] != keep]:
            DISABLED[name] = "meta가 다른 방법들과 불일치 (다른 데이터로 빌드됨)"
            del METHODS[name]

    _load_backbones(device)
    canon = next(iter(METHODS.values()))
    meta = canon["meta"]

    import json as _json
    labels = sorted({r.get("text", "").strip() for r in meta if r.get("text", "").strip()})
    loc_path = os.path.join(ROOT, "data/locarno_9.json")
    loc = _json.load(open(loc_path, encoding="utf-8")) if os.path.exists(loc_path) \
        else {"subclasses": {}, "classes": {}}

    # 출원(patentID)별 전체 도면 맵 - 팝업용. 인덱스가 아니라 전체 pairs.jsonl 사용.
    from collections import defaultdict
    all_records = load_records(os.path.join(ROOT, "data/pairs.jsonl"))
    design_map = defaultdict(list)
    for gi, r in enumerate(all_records):
        design_map[r.get("design_id", "")].append(gi)

    STATE.update(device=device, meta=meta, labels=labels, locarno=loc,
                 all_records=all_records, design_map=design_map,
                 default=next(iter(METHODS)))
    for name, m in METHODS.items():
        print(f"[{name}] {m['ntotal']} vectors, {m['size']}px")
    for name, why in DISABLED.items():
        print(f"[{name}] 비활성: {why}")
    print(f"default={STATE['default']}, labels={len(labels)}, device={device}")


@app.on_event("startup")
def _startup():
    load()


@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8").read()


@app.get("/api/methods")
def methods():
    return JSONResponse({
        "default": STATE["default"],
        "methods": [{"name": n, "desc": m["desc"], "ntotal": m["ntotal"],
                     "image_size": m["size"]} for n, m in METHODS.items()],
        "disabled": DISABLED,
    })


def _resolve_method(method):
    name = (method or "").strip() or STATE["default"]
    if name not in METHODS:
        raise HTTPException(400, f"알 수 없는 방법: {name} (가능: {list(METHODS)})")
    return name


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


def _group_pack(idxs, scores, topk, agg_top=2):
    """출원(design_id)별로 풀 내 뷰 점수를 모아 상위 agg_top개 평균으로 집계 후 topk.

    최고 뷰 1장(max)만 쓰면 노이즈에 취약 → top-2 뷰 평균이 강건 (ACCURACY.md ⑥).
    대표 이미지는 해당 출원의 최고 점수 뷰."""
    views, best = {}, {}
    for i, s in zip(np.asarray(idxs).tolist(), np.asarray(scores).tolist()):
        if i < 0:
            continue
        did = STATE["meta"][int(i)].get("design_id", "")
        views.setdefault(did, []).append(s)
        if did not in best or s > best[did][1]:
            best[did] = (int(i), s)
    ranked = sorted(
        ((float(np.mean(sorted(v, reverse=True)[:agg_top])), d) for d, v in views.items()),
        reverse=True)
    return [_pack_one(best[d][0], agg) for agg, d in ranked[:topk]]


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


def _locarno_facets_pool(method, qvec, topn=10, pool=200):
    """유사도 상위 pool개 도면에서 로카르노를 '유사도 가중합'으로 랭킹 → top-N."""
    idx = METHODS[method]["index"]
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
        a[0] += max(0.0, float(s))
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


def _label_vecs(method):
    """이미지→명칭(zero-shot) 라벨 뱅크. 텍스트 인코더가 방법마다 다르므로 방법별로
    지연 인코딩해 캐시한다 (첫 이미지 검색에서 수 초)."""
    if method not in LABEL_CACHE:
        model, proc, _, ctx = _use(method)
        with ctx:
            LABEL_CACHE[method] = encode_text(model, proc, STATE["labels"], STATE["device"])
    return LABEL_CACHE[method]


def _predict_labels(method, img_vec, topn=5):
    sims = (_label_vecs(method) @ img_vec[0])           # [L]
    top = np.argsort(-sims)[:topn]
    return [{"label": STATE["labels"][j], "score": round(float(sims[j]), 4)} for j in top]


def _reconstruct(index, ids):
    """필요한 행만 인덱스에서 복원 - 전체 벡터 복사본(방법당 0.4GB)을 상주시키지 않는다."""
    return index.reconstruct_batch(np.asarray(ids, dtype="int64"))


def _encode_image_query(method, img, tta=True):
    """쿼리 이미지 인코딩. TTA: 원본+소회전(±4°) 앙상블 평균 → 재정규화 (ACCURACY.md ⑧)."""
    imgs = [img]
    if tta:
        imgs += [img.rotate(a, expand=True, fillcolor=(255, 255, 255)) for a in (-4, 4)]
    model, proc, size, ctx = _use(method)
    with ctx:
        vecs = encode_pil(model, proc, imgs, size, STATE["device"])
    v = vecs.mean(0, keepdims=True)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")


def _encode_text_query(method, q):
    """텍스트 쿼리: 프롬프트 템플릿 앙상블(한글 감지 시 한국어 템플릿) 평균 → 재정규화."""
    if any("가" <= ch <= "힣" for ch in q):
        tpls = [q, f"{q}의 특허 도면", f"{q} 디자인 도면"]
    else:
        tpls = [q, f"patent drawing of {q}", f"technical line drawing of {q}"]
    model, proc, _, ctx = _use(method)
    with ctx:
        vecs = encode_text(model, proc, tpls, STATE["device"])
    v = vecs.mean(0, keepdims=True)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")


def _query_expand(method, vec, m=10, alpha=3.0):
    """α-쿼리 확장(αQE): 1차 검색 top-m 이미지 벡터를 점수^α 가중으로 쿼리에 더해
    재검색. 단일 뷰 쿼리를 갤러리 분포 쪽으로 보강하는 경량 re-rank (ACCURACY.md ⑦)."""
    index = METHODS[method]["index"]
    s, i = index.search(vec.astype("float32"), m)
    keep = i[0] >= 0
    if not keep.any():
        return vec
    nb = _reconstruct(index, i[0][keep])
    w = np.clip(s[0][keep], 0.0, None) ** alpha
    v = vec[0] + (nb * w[:, None]).sum(0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v[None].astype("float32")


def _search_image_one(method, img, topk, category, alpha, tta, qe):
    """한 방법으로 이미지 검색. (payload dict) 반환. _lock 안에서 호출."""
    index = METHODS[method]["index"]
    img_vec = _encode_image_query(method, img, tta=tta)
    predicted = _predict_labels(method, img_vec)
    used_cat = category.strip() or (predicted[0]["label"] if predicted else "")
    qvec = _query_expand(method, img_vec) if qe else img_vec

    if alpha < 1.0 and used_cat:
        # 개념 검색(image→text→image): ①카테고리 텍스트로 후보 풀 확정 → ②풀 내에서
        # 이미지 유사도로 재정렬. 최종점수 = alpha*이미지유사 + (1-alpha)*카테고리유사.
        tvec = _encode_text_query(method, used_cat)
        pool = min(max(int(topk) * 25, 200), index.ntotal)
        t_scores, t_idxs = index.search(tvec, pool)
        ids, tsc = t_idxs[0], t_scores[0]
        keep = ids >= 0
        ids, tsc = ids[keep], tsc[keep]
        isc = _reconstruct(index, ids) @ qvec[0]
        final = alpha * isc + (1.0 - alpha) * tsc
        order = np.argsort(-final)
        sel_ids, sel_sc = ids[order], final[order]
    else:
        used_cat = ""
        pool = min(max(int(topk) * 12, 120), index.ntotal)
        sel_sc, sel_ids = index.search(qvec, pool)
        sel_ids, sel_sc = sel_ids[0], sel_sc[0]

    facets = _locarno_facets_pool(method, img_vec)
    results = _group_pack(sel_ids, sel_sc, int(topk))
    return {"method": method, "count": len(results),
            "predicted_labels": predicted, "used_category": used_cat, "alpha": alpha,
            "results": results, "locarno_facets": facets}


def _search_text_one(method, query, topk):
    index = METHODS[method]["index"]
    vec = _encode_text_query(method, query)
    pool = min(max(int(topk) * 12, 120), index.ntotal)
    scores, idxs = index.search(vec, pool)
    facets = _locarno_facets_pool(method, vec)
    results = _group_pack(idxs[0], scores[0], int(topk))
    return {"method": method, "count": len(results), "query": query,
            "results": results, "locarno_facets": facets}


def _union(payloads):
    """방법별 topk 결과의 합집합. 출원 단위로 어느 방법이 몇 위에 올렸는지 (스펙 5장)."""
    designs = {}
    for p in payloads:
        for rank, r in enumerate(p["results"], 1):
            d = designs.setdefault(r["design_id"], {
                "design_id": r["design_id"], "patent_no": r["patent_no"],
                "title": r["title"], "image_url": r["image_url"], "picks": {}})
            d["picks"][p["method"]] = rank
    ranked = sorted(designs.values(),
                    key=lambda d: (-len(d["picks"]), min(d["picks"].values())))
    return ranked


def _parse_methods(method, methods_csv):
    """단일 method 또는 콤마 목록 methods → 검증된 이름 리스트."""
    if methods_csv.strip():
        names = [_resolve_method(x) for x in methods_csv.split(",") if x.strip()]
        return list(dict.fromkeys(names))[:8] or [_resolve_method(method)]
    return [_resolve_method(method)]


@app.post("/api/search")
def search(image: UploadFile = File(...), topk: int = 12,
           category: str = Form(""), alpha: float = Form(1.0),
           tta: int = Form(1), qe: int = Form(1),
           method: str = Form(""), methods: str = Form("")):
    """이미지 검색. method= 단일 방법, methods=a,b 면 나란히 비교."""
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")
    names = _parse_methods(method, methods)
    alpha = max(0.0, min(1.0, float(alpha)))
    with _lock:
        payloads = [_search_image_one(n, img, topk, category, alpha, bool(tta), bool(qe))
                    for n in names]
    if len(payloads) == 1:
        return JSONResponse(payloads[0])
    return JSONResponse({"compare": True, "methods": payloads, "union": _union(payloads)})


@app.post("/api/search_text")
def search_text(query: str = Form(...), topk: int = 12,
                method: str = Form(""), methods: str = Form("")):
    query = query.strip()
    if not query:
        raise HTTPException(400, "검색어를 입력하세요")
    names = _parse_methods(method, methods)
    with _lock:
        payloads = [_search_text_one(n, query, topk) for n in names]
    if len(payloads) == 1:
        return JSONResponse(payloads[0])
    return JSONResponse({"compare": True, "methods": payloads, "union": _union(payloads)})


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
    # PORT 환경변수로 오버라이드 가능 - 8000이 다른 프로세스에 선점된 채로 띄우면
    # uvicorn이 bind 실패를 로그에만 남기고, 사용자는 그 포트의 엉뚱한 서버를 보게 된다.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
