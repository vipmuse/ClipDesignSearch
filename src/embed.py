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

from dataset import load_records, preprocess_drawing, take_limit
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_posix(path):
    """저장소 기준 상대 POSIX 경로로 정규화. 저장소 밖 경로는 그대로 둔다.

    지문에 머신 절대경로가 박히면 저장소를 복제·이동한 순간 아무것도 바뀌지 않았는데
    전부 '불일치'가 된다. 반대로 서버는 resolved config의 train.output_dir(규약상 항상
    상대경로)에서 어댑터 경로를 유도하므로, 양쪽 표기가 다르면 영원히 만나지 못한다.
    호출자가 절대/상대 어느 쪽을 넘기든 같은 문자열이 되도록 여기서 통일한다.
    """
    s = str(path)
    if os.path.isabs(s):
        try:
            rel = os.path.relpath(s, ROOT)
        except ValueError:                 # 다른 드라이브 → 상대화 불가
            rel = ".."
        if not rel.startswith(".."):       # 저장소 안일 때만 상대화
            s = rel
    return s.replace("\\", "/")


def index_fingerprint(cfg, adapter, data, limit=0):
    """인덱스가 어떤 모델·어댑터·데이터·범위로 만들어졌는지 기록하는 지문.

    방법이 여러 개면 'A 인덱스에 B 어댑터'로 검색하는 사고가 실제로 일어나고,
    그건 에러 없이 결과만 조용히 틀어진다. encode_images의 ckpt_key와 같은 원칙.

    limit이 지문에 있어야 하는 이유: --quick(--limit 2000)이 만든 2,000장짜리
    데모 인덱스와 472,615장 전체 인덱스는 model_id·adapter·data가 전부 같다.
    limit이 빠지면 서버가 데모 인덱스를 전체 갤러리로 믿고 서빙한다.
    """
    return {
        "model_id": cfg["model"]["model_id"],
        "image_size": cfg["model"]["image_size"],
        "adapter": rel_posix(adapter),
        "data": rel_posix(data),
        "limit": int(limit or 0),
        "method": cfg.get("method", {}).get("name", ""),
    }


def check_index_meta(index_dir, cfg, adapter, data, limit=0, n_vectors=None):
    """인덱스 지문이 현재 설정과 맞는지 확인. (정상여부, 사유) 반환.

    서버는 이 결과로 방법별 활성 여부를 정한다 — 불일치 시 그 방법만 비활성시키고
    서버 전체를 죽이지 않는다.

    n_vectors는 지문(=입력)이 아니라 빌드 결과라 want에 넣을 수 없다(열리지 않는
    도면이 몇 장이었는지를 요청 측이 미리 알 도리가 없다). 대신 실제 벡터 수를
    아는 호출자(FAISS 인덱스를 연 서버: index.ntotal)가 넘기면 따로 대조한다 —
    faiss.index만 잘리거나 뒤바뀐 경우를 잡는 용도.
    """
    p = os.path.join(index_dir, "index_meta.json")
    if not os.path.exists(p):
        return False, f"index_meta.json 없음: {p}"
    with open(p, encoding="utf-8") as f:   # 서버가 요청마다 부르는 함수 → fd 누수 금지
        saved = json.load(f)
    want = index_fingerprint(cfg, adapter, data, limit)
    for k, v in want.items():
        if k == "method":                      # 이름은 참고용, 판정에 쓰지 않는다
            continue
        if saved.get(k) != v:
            return False, f"{k} 불일치: 인덱스={saved.get(k)!r} 요청={v!r}"
    if n_vectors is not None and saved.get("n_vectors") != int(n_vectors):
        return False, f"n_vectors 불일치: 인덱스={saved.get('n_vectors')!r} 실제={int(n_vectors)!r}"
    return True, ""


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
def encode_text(model, proc, texts, device, batch_size=256):
    """텍스트 리스트 → L2 정규화 임베딩 [N, D]. 이미지 임베딩과 같은 joint 공간.
    라벨 뱅크(수만 개)를 한 번에 넣으면 VRAM 폭발 → 청크 단위로 배치 처리."""
    texts = list(texts)
    out = []
    for k in range(0, len(texts), batch_size):
        tok = proc(text=texts[k:k + batch_size], return_tensors="pt", padding=True,
                   truncation=True, max_length=77).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            emb = model.get_text_features(**tok)
        if not torch.is_tensor(emb):       # MetaCLIP 2는 출력 객체 반환 → pooler_output 사용
            emb = emb.pooler_output
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        out.append(emb.cpu().numpy().astype("float32"))
    return np.concatenate(out) if out else np.zeros((0, 1), "float32")


@torch.no_grad()
def encode_images(model, proc, records, image_root, size, device, batch_size=64,
                  ckpt_dir=None, ckpt_every=20000, ckpt_key=None):
    """레코드들을 배치로 인코딩 (GPU 효율). 열 수 없는 이미지는 건너뜀.
    ckpt_dir 지정 시 ckpt_every장마다 중간 저장 → 중단돼도 이어서 재개.
    ckpt_key(모델/데이터 식별자)가 다르면 남은 체크포인트를 버리고 처음부터 —
    다른 어댑터·데이터로 만든 벡터가 섞여 조용히 오염된 인덱스가 되는 것을 방지."""
    from tqdm import tqdm
    Image.MAX_IMAGE_PIXELS = None          # 초대형 도면의 DecompressionBomb 예외 방지
    vecs, kept = [], []
    buf, buf_idx = [], []
    start = 0

    emb_p = kept_p = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        emb_p = os.path.join(ckpt_dir, "ckpt_emb.npy")
        kept_p = os.path.join(ckpt_dir, "ckpt_kept.json")
        if os.path.exists(emb_p) and os.path.exists(kept_p):
            with open(kept_p, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("key") != ckpt_key:                 # 어댑터/데이터/limit 불일치
                print("[ckpt] 설정이 달라 기존 체크포인트를 폐기하고 처음부터", flush=True)
            else:
                vecs.append(np.load(emb_p))
                kept = saved["kept"]
                start = (max(kept) + 1) if kept else 0
                print(f"[ckpt] {len(kept)}장 복원, {start}번부터 재개", flush=True)

    def save_ckpt():
        if not emb_p:
            return
        mat = np.concatenate(vecs).astype("float32")
        np.save(emb_p + ".tmp.npy", mat)                     # 저장 중 사망해도 원본 보존
        os.replace(emb_p + ".tmp.npy", emb_p)
        with open(kept_p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"key": ckpt_key, "kept": kept}, f)
        os.replace(kept_p + ".tmp", kept_p)
        vecs.clear(); vecs.append(mat)                       # 조각 누적 방지

    def flush():
        if buf:
            vecs.append(encode_pil(model, proc, buf, size, device))
            kept.extend(buf_idx)
            buf.clear(); buf_idx.clear()

    last_saved = len(kept)
    for i, r in enumerate(tqdm(records[start:], desc="encoding", initial=start,
                               total=len(records)), start=start):
        try:                                # 로딩·디코딩 오류를 개별 격리
            im = Image.open(os.path.join(image_root, r["image"]))
            im.load()
            buf.append(im.convert("RGB"))
            buf_idx.append(i)
        except Exception:
            continue
        if len(buf) >= batch_size:
            flush()
            if ckpt_dir and len(kept) - last_saved >= ckpt_every:
                save_ckpt()
                last_saved = len(kept)
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
    ap.add_argument("--limit", type=int, default=0,
                    help="build: N개 도면만 인덱싱(데모용). "
                         "train/eval과 같은 seed로 무작위 표집(앞에서 자르지 않음)")
    args = ap.parse_args()

    import faiss
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["model"]["image_size"]
    model, proc = load_tuned(cfg, args.adapter)
    model.to(device)

    if args.cmd == "build":
        records = load_records(args.data)
        # 앞에서 자르면(contiguous slice) 인제스천 순서가 몰린 좁은 구간만 인덱싱된다.
        # train/eval과 같은 seed로 take_limit을 써야 세 산출물이 같은 레코드 집합을 본다.
        records = take_limit(records, args.limit, cfg["train"]["seed"])
        ckpt_dir = os.path.join(args.index, "ckpt")
        ckpt_key = {                              # 재개 가능 여부를 가르는 식별자
            "model_id": cfg["model"]["model_id"], "image_size": size,
            "adapter": str(args.adapter), "data": args.data,
            "image_root": args.image_root, "limit": args.limit,
            "n_records": len(records),
        }
        mat, kept = encode_images(model, proc, records, args.image_root, size, device,
                                  ckpt_dir=ckpt_dir, ckpt_key=ckpt_key)
        records = [records[i] for i in kept]      # 인코딩 성공한 레코드만 meta에 저장
        index = faiss.IndexFlatIP(mat.shape[1])   # 정규화 벡터 → 내적 = 코사인
        index.add(mat)
        os.makedirs(args.index, exist_ok=True)
        faiss.write_index(index, os.path.join(args.index, "faiss.index"))
        with open(os.path.join(args.index, "meta.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(args.index, "index_meta.json"), "w", encoding="utf-8") as f:
            fp = index_fingerprint(cfg, args.adapter, args.data, args.limit)
            fp.update(n_vectors=int(mat.shape[0]), dim=int(mat.shape[1]))
            json.dump(fp, f, ensure_ascii=False, indent=2)
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)   # 완료 후 체크포인트 정리
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
