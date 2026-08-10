"""전체 갤러리 리트리벌 평가: design_id 분할 홀드아웃에서 I→I / T→I R@K + mAP.

train.py의 배치 내 Recall(후보 32개)은 변별력이 낮은 프록시 — 공식 수치는 이 스크립트.
train과 같은 --eval-ratio/--seed(기본: config 값)를 쓰면 같은 홀드아웃을 본다.

  # 베이스라인 (튜닝 전)
  python src/eval_retrieval.py --adapter none
  # 튜닝 후 → 베이스라인과 비교
  python src/eval_retrieval.py --adapter outputs/lora-clip-design/final

정의:
  I→I: 각 홀드아웃 도면이 쿼리, 나머지 전부가 갤러리. 같은 design_id의 다른 뷰가 정답.
  T→I: 홀드아웃의 유니크 텍스트가 쿼리. 그 텍스트를 가진 모든 도면이 정답.
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from dataset import load_records, split_by_design
from embed import encode_images, encode_text, load_tuned
from metrics import MetricAccumulator, rank_metrics

CHUNK = 256          # 쿼리 청크 (메모리 절약: sim 행렬을 [CHUNK, N]씩 계산)


def eval_i2i(img_mat, design_ids):
    """도면→도면: 같은 design_id(자기 제외)가 정답."""
    acc = MetricAccumulator()
    d = np.asarray(design_ids)
    N = img_mat.shape[0]
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        sim = img_mat[s:e] @ img_mat.T                   # [q, N]
        rel = d[s:e, None] == d[None, :]
        rows = np.arange(e - s)
        sim[rows, np.arange(s, e)] = -np.inf             # 자기 자신 제외
        rel[rows, np.arange(s, e)] = False
        acc.update(*rank_metrics(sim, rel))
    return acc.result()


def eval_t2i(txt_mat, uniq_texts, img_mat, img_texts):
    """텍스트→도면: 동일 텍스트를 가진 모든 도면이 정답."""
    acc = MetricAccumulator()
    it = np.asarray(img_texts)
    for s in range(0, len(uniq_texts), CHUNK):
        e = min(s + CHUNK, len(uniq_texts))
        sim = txt_mat[s:e] @ img_mat.T
        rel = np.asarray(uniq_texts[s:e])[:, None] == it[None, :]
        acc.update(*rank_metrics(sim, rel))
    return acc.result()


def encode_text_chunked(model, proc, texts, device, bs=256):
    return np.concatenate([encode_text(model, proc, texts[i:i + bs], device)
                           for i in range(0, len(texts), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--adapter", required=True, help="어댑터 경로 또는 'none'(베이스라인)")
    ap.add_argument("--data", default=None, help="기본: config의 data_path")
    ap.add_argument("--image-root", default="data")
    ap.add_argument("--eval-ratio", type=float, default=0, help="기본: config eval_ratio")
    ap.add_argument("--seed", type=int, default=0, help="기본: config seed")
    ap.add_argument("--limit", type=int, default=0, help="스모크: 전체 레코드 앞 N개만")
    ap.add_argument("--out", default="outputs/eval", help="결과 JSON 저장 디렉터리")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    ratio = args.eval_ratio or t["eval_ratio"]
    seed = args.seed or t["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_records(args.data or t["data_path"])
    if args.limit:
        records = records[:args.limit]
    _, eval_recs = split_by_design(records, ratio, seed)   # train.py와 동일 분할
    print(f"holdout: {len(eval_recs)} records "
          f"({len(set(r.get('design_id') for r in eval_recs))} designs)")

    model, proc = load_tuned(cfg, args.adapter)
    model.to(device)

    img_mat, kept = encode_images(model, proc, eval_recs, args.image_root,
                                  cfg["model"]["image_size"], device)
    eval_recs = [eval_recs[i] for i in kept]
    design_ids = [r.get("design_id", r["image"]) for r in eval_recs]
    img_texts = [r["text"] for r in eval_recs]

    i2i = eval_i2i(img_mat, design_ids)
    print(f"I->I  {i2i}")

    uniq_texts = sorted(set(img_texts))
    txt_mat = encode_text_chunked(model, proc, uniq_texts, device)
    t2i = eval_t2i(txt_mat, uniq_texts, img_mat, img_texts)
    print(f"T->I  {t2i}")

    name = "base" if str(args.adapter).lower() in ("none", "", "base") \
        else os.path.basename(os.path.normpath(args.adapter))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"adapter": args.adapter, "gallery": len(eval_recs),
                   "eval_ratio": ratio, "seed": seed,
                   "I2I": i2i, "T2I": t2i}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
