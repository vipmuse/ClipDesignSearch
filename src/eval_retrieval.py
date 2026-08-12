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

from dataset import load_records, split_by_design, take_limit
from embed import encode_images, encode_text, load_tuned
from metrics import MetricAccumulator, rank_metrics
from vacsr import load_adapter, pairwise_score

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


def eval_t2i(txt_mat, uniq_texts, img_mat, img_texts, scorer=None):
    """텍스트→도면: 동일 텍스트를 가진 모든 도면이 정답.

    scorer가 있으면(vacsr arm) 코사인 대신 그 함수로 유사도를 계산한다 -
    vacsr의 유사도는 어댑터 출력이라 내적이 아니고, 코사인으로 평가하면
    학습된 것과 다른 것을 재게 된다. scorer(s, e) -> [e-s, N] numpy.
    """
    acc = MetricAccumulator()
    it = np.asarray(img_texts)
    for s in range(0, len(uniq_texts), CHUNK):
        e = min(s + CHUNK, len(uniq_texts))
        sim = scorer(s, e) if scorer is not None else txt_mat[s:e] @ img_mat.T
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
    ap.add_argument("--eval-ratio", type=float, default=None, help="기본: config eval_ratio")
    ap.add_argument("--seed", type=int, default=None, help="기본: config seed")
    ap.add_argument("--limit", type=int, default=0, help="스모크: 레코드 N개만 사용")
    ap.add_argument("--out", default="outputs/eval", help="결과 JSON 저장 디렉터리")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    ratio = args.eval_ratio if args.eval_ratio is not None else t["eval_ratio"]
    seed = args.seed if args.seed is not None else t["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_records(args.data or t["data_path"])
    records = take_limit(records, args.limit, seed)        # train.py와 동일 seed → 동일 표본
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

    # vacsr 어댑터가 체크포인트에 있으면 T2I는 어댑터 점수로 잰다. I2I는 크로스모달
    # 어댑터의 적용 대상이 아니라 코사인 유지 - 방법의 정의가 그렇다 (src/vacsr.py).
    vac = None if str(args.adapter).lower() in ("none", "", "base")         else load_adapter(args.adapter, device)
    scorer = None
    if vac is not None:
        txt_t = torch.from_numpy(txt_mat).to(device)
        img_t = torch.from_numpy(img_mat).to(device)
        scorer = lambda s, e: pairwise_score(vac, txt_t[s:e], img_t).cpu().numpy()
        print("[vacsr] T2I를 어댑터 점수로 평가")
    t2i = eval_t2i(txt_mat, uniq_texts, img_mat, img_texts, scorer=scorer)
    print(f"T->I  {t2i}")

    name = "base" if str(args.adapter).lower() in ("none", "", "base") \
        else os.path.basename(os.path.normpath(args.adapter))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"adapter": args.adapter, "gallery": len(eval_recs),
                   "eval_ratio": ratio, "seed": seed,
                   "t2i_scorer": "vacsr" if vac is not None else "cosine",
                   "I2I": i2i, "T2I": t2i}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
