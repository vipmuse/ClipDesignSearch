"""정답 라벨 없이 도면 검색 성능 평가 (held-out 연도, 예: 2019).

정답 신호(데이터 내재):
  - 같은 출원(design_id)의 다른 도면 = 같은 디자인의 다른 뷰 → "정답 유사 이미지"
  - 같은 로카르노(과) = 같은 물품 카테고리

측정 지표:
  A. 같은-출원 검색: Recall@1/5/10, mAP  (relevant = 같은 design_id, 자기 제외)
  B. 같은-로카르노 검색: P@10, mAP       (relevant = 같은 locarno)
  C. 분류 top10 정확도: 유사도 가중 top1/top3 로카르노가 쿼리의 실제 로카르노와 일치하는가

  # 학습된 모델 평가
  python src/evaluate_retrieval.py --adapter outputs/lora-clip-design/final --data data/eval_2019.jsonl --image-root .
  # 학습 전(baseline) 비교
  python src/evaluate_retrieval.py --base --data data/eval_2019.jsonl --image-root .
"""
import argparse
import os
import random
import sys

import faiss
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_records                       # noqa: E402
from embed import encode_images, load_tuned            # noqa: E402


def build_model(cfg, adapter, use_base):
    """use_base=True면 학습 전 MetaCLIP2, 아니면 LoRA 어댑터 병합 모델."""
    if use_base:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(cfg["model"]["model_id"], attn_implementation="sdpa")
        proc = AutoProcessor.from_pretrained(cfg["model"]["model_id"])
        model.eval()
        return model, proc
    return load_tuned(cfg, adapter)


def subsample_by_design(records, n_designs, seed=42):
    """디자인(design_id) 단위로 N개 샘플 + 그 도면 전체. 같은-출원 평가를 위해 뷰가 여러 개 유지됨."""
    by = {}
    for r in records:
        by.setdefault(r.get("design_id", ""), []).append(r)
    ids = list(by)
    random.Random(seed).shuffle(ids)
    if n_designs:
        ids = ids[:n_designs]
    out = []
    for i in ids:
        out.extend(by[i])
    return out


def evaluate(emb, meta, topk=200, k_facet=50):
    N = emb.shape[0]
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    K = min(topk + 1, N)
    sims, idxs = index.search(emb, K)                   # 자기 자신 포함

    dids = [m.get("design_id", "") for m in meta]
    locs = [m.get("locarno", "") for m in meta]
    # design_id별 개수 (같은-출원 정답 수 = 자기 제외)
    from collections import Counter
    dcount = Counter(dids)
    lcount = Counter(l for l in locs if l)              # 로카르노별 전체 개수 (mAP 분모용)

    r1 = r5 = r10 = 0
    ap_design, ap_loc, p10_loc = [], [], []
    fac1 = fac3 = fac_total = 0
    n_design_q = 0                                      # 같은-출원 파트너가 있는 쿼리 수

    for i in range(N):
        # 자기 자신 제거한 이웃 순위
        neigh = [(int(j), float(s)) for j, s in zip(idxs[i], sims[i]) if int(j) != i]
        neigh = neigh[:topk]
        n_ids = [j for j, _ in neigh]

        # ---- A. 같은-출원 ----
        n_rel_design = dcount[dids[i]] - 1
        if n_rel_design > 0:
            n_design_q += 1
            hits = [dids[j] == dids[i] for j in n_ids]
            if any(hits[:1]):
                r1 += 1
            if any(hits[:5]):
                r5 += 1
            if any(hits[:10]):
                r10 += 1
            # AP
            hit_cnt = 0
            prec_sum = 0.0
            for rank, h in enumerate(hits, 1):
                if h:
                    hit_cnt += 1
                    prec_sum += hit_cnt / rank
            ap_design.append(prec_sum / min(n_rel_design, topk))

        # ---- B. 같은-로카르노 ----
        if locs[i]:
            lhits = [locs[j] == locs[i] for j in n_ids]
            p10_loc.append(sum(lhits[:10]) / 10.0)
            n_rel_loc = lcount[locs[i]] - 1             # 같은 로카르노 개수(자기 제외), O(1)
            if n_rel_loc > 0:
                hit_cnt = 0
                prec_sum = 0.0
                for rank, h in enumerate(lhits, 1):
                    if h:
                        hit_cnt += 1
                        prec_sum += hit_cnt / rank
                ap_loc.append(prec_sum / min(n_rel_loc, topk))

        # ---- C. 분류 top10 (유사도 가중) 정확도 ----
        if locs[i]:
            agg = {}
            for j, s in neigh[:k_facet]:
                if locs[j]:
                    agg[locs[j]] = agg.get(locs[j], 0.0) + max(0.0, s)
            if agg:
                ranked = sorted(agg, key=agg.get, reverse=True)
                fac_total += 1
                if ranked[0] == locs[i]:
                    fac1 += 1
                if locs[i] in ranked[:3]:
                    fac3 += 1

    d = max(1, n_design_q)
    return {
        "queries": N, "same_design_queries": n_design_q,
        "A_same_design": {
            "R@1": r1 / d, "R@5": r5 / d, "R@10": r10 / d,
            "mAP": float(np.mean(ap_design)) if ap_design else 0.0,
        },
        "B_same_locarno": {
            "P@10": float(np.mean(p10_loc)) if p10_loc else 0.0,
            "mAP": float(np.mean(ap_loc)) if ap_loc else 0.0,
        },
        "C_locarno_facet": {
            "top1_acc": fac1 / max(1, fac_total),
            "top3_acc": fac3 / max(1, fac_total),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--adapter", default="outputs/lora-clip-design/final")
    ap.add_argument("--base", action="store_true", help="학습 전 baseline 평가")
    ap.add_argument("--data", default="data/eval_2019.jsonl")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--designs", type=int, default=3000, help="평가에 쓸 디자인 수(0=전체)")
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["model"]["image_size"]

    records = subsample_by_design(load_records(args.data), args.designs)
    tag = "BASELINE(학습전)" if args.base else "TRAINED(학습후)"
    print(f"[{tag}] 평가 도면 {len(records)}장, "
          f"고유 디자인 {len({r.get('design_id','') for r in records})}개")

    model, proc = build_model(cfg, args.adapter, args.base)
    model.to(device)
    emb, kept = encode_images(model, proc, records, args.image_root, size, device)
    meta = [records[i] for i in kept]

    import json
    res = evaluate(emb, meta, topk=args.topk)
    print(f"\n=== [{tag}] 평가 결과 ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
