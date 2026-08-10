"""추론: 학습된 LoRA 어댑터로 도면 DB 인코딩 + FAISS 검색.

  # 인덱스 구축
  python src/embed.py build --adapter outputs/lora-clip-design/final --data data/pairs.jsonl
  # 텍스트로 검색
  python src/embed.py search --adapter outputs/lora-clip-design/final --text "무선 이어폰 케이스"
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModel, AutoProcessor

from dataset import load_records, preprocess_drawing
from PIL import Image


def load_tuned(cfg, adapter_dir, merge=True):
    """adapter_dir이 None/'none'이면 베이스 모델 그대로 (튜닝 전 베이스라인 평가용)."""
    base = AutoModel.from_pretrained(cfg["model"]["model_id"], attn_implementation="sdpa")
    if adapter_dir and str(adapter_dir).lower() not in ("none", "base"):
        model = PeftModel.from_pretrained(base, adapter_dir)
        if merge:
            model = model.merge_and_unload()   # 어댑터를 백본에 병합 → 추론 지연 0
    else:
        model = base
    model.eval()
    proc = AutoProcessor.from_pretrained(cfg["model"]["model_id"])
    return model, proc


@torch.no_grad()
def encode_pil(model, proc, imgs, size, device):
    """PIL 이미지 리스트 → L2 정규화 임베딩 [N, D] (numpy float32)."""
    imgs = [preprocess_drawing(im, size) for im in imgs]
    px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        emb = model.get_image_features(pixel_values=px)
    # MetaCLIP 2는 텐서가 아니라 출력 객체를 반환 → pooler_output(=투영된 이미지 임베딩, 1024) 사용
    if not torch.is_tensor(emb):
        emb = emb.pooler_output
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy().astype("float32")


@torch.no_grad()
def encode_text(model, proc, texts, device):
    """텍스트 리스트 → L2 정규화 임베딩 [N, D]. 이미지 임베딩과 같은 joint 공간."""
    tok = proc(text=list(texts), return_tensors="pt", padding=True,
               truncation=True, max_length=77).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        emb = model.get_text_features(**tok)
    if not torch.is_tensor(emb):           # MetaCLIP 2는 출력 객체 반환 → pooler_output 사용
        emb = emb.pooler_output
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy().astype("float32")


@torch.no_grad()
def encode_images(model, proc, records, image_root, size, device, batch_size=64):
    """레코드들을 배치로 인코딩 (GPU 효율). 열 수 없는 이미지는 건너뜀."""
    from tqdm import tqdm
    Image.MAX_IMAGE_PIXELS = None          # 초대형 도면의 DecompressionBomb 예외 방지
    vecs, kept = [], []
    buf, buf_idx = [], []

    def flush():
        if buf:
            vecs.append(encode_pil(model, proc, buf, size, device))
            kept.extend(buf_idx)
            buf.clear(); buf_idx.clear()

    for i, r in enumerate(tqdm(records, desc="encoding")):
        try:                                # 로딩·디코딩 오류를 개별 격리
            im = Image.open(os.path.join(image_root, r["image"]))
            im.load()
            buf.append(im.convert("RGB"))
            buf_idx.append(i)
        except Exception:
            continue
        if len(buf) >= batch_size:
            flush()
    flush()
    mat = np.concatenate(vecs).astype("float32") if vecs else np.zeros((0, 1), "float32")
    return mat, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/pairs.jsonl")
    ap.add_argument("--image-root", default="data")
    ap.add_argument("--index", default="outputs/index")
    ap.add_argument("--text", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="build: 앞 N개 도면만 인덱싱(데모용)")
    args = ap.parse_args()

    import faiss
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["model"]["image_size"]
    model, proc = load_tuned(cfg, args.adapter)
    model.to(device)

    if args.cmd == "build":
        records = load_records(args.data)
        if args.limit:
            records = records[:args.limit]
        mat, kept = encode_images(model, proc, records, args.image_root, size, device)
        records = [records[i] for i in kept]      # 인코딩 성공한 레코드만 meta에 저장
        index = faiss.IndexFlatIP(mat.shape[1])   # 정규화 벡터 → 내적 = 코사인
        index.add(mat)
        os.makedirs(args.index, exist_ok=True)
        faiss.write_index(index, os.path.join(args.index, "faiss.index"))
        with open(os.path.join(args.index, "meta.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"built index: {mat.shape[0]} vectors -> {args.index}")

    elif args.cmd == "search":
        assert args.text, "--text 쿼리를 주세요"
        index = faiss.read_index(os.path.join(args.index, "faiss.index"))
        meta = load_records(os.path.join(args.index, "meta.jsonl"))
        q = encode_text(model, proc, [args.text], device)
        scores, idxs = index.search(q, args.topk)
        for rank, (s, i) in enumerate(zip(scores[0], idxs[0]), 1):
            print(f"{rank:2d}. {s:.3f}  {meta[i]['image']}  {meta[i].get('text','')}")


if __name__ == "__main__":
    main()
