"""대조학습 루프: CLIP 이미지↔텍스트 + 이미지↔이미지 supervised contrastive.

정확도 개선 사항(ACCURACY.md):
  - design_id 단위 train/eval 분할 (평가 누수 제거)
  - PK 배치 샘플러: 배치 = 디자인 P개 × 뷰 K장 → img2img loss가 실제 발화
  - masked InfoNCE: 같은 design_id의 다른 뷰를 네거티브로 밀어내는 노이즈 제거
    (텍스트 동일성은 positive 근거에서 제외 — 제목 중복이 심해 마스크가 포화된다)
  - 학습 시 도면 증강 (소회전·스케일·라인두께)

사용법:
  python src/train.py --config configs/lora_clip.yaml
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import Collator, PairDataset, PKBatchSampler, load_records, split_by_design, take_limit
from hobit import HobitBatchSampler, refresh_embeddings
from model import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def masked_clip_loss(logits_per_image, pos_mask):
    """멀티-positive InfoNCE (양방향 평균).

    pos_mask[i,j]=True ⇔ (이미지 i, 텍스트 j)가 정답 쌍. 대각선 외에도 같은
    design_id의 다른 뷰를 positive로 인정해, 표준 CLIP loss가 이들을 네거티브로
    밀어내는 false negative 노이즈를 제거한다. 텍스트 동일성은 근거에 넣지 않는다
    (이유는 아래 _pos_mask 참조)."""
    m = pos_mask.float()
    logp_i2t = logits_per_image.log_softmax(dim=1)      # 이미지→텍스트 방향
    logp_t2i = logits_per_image.log_softmax(dim=0)      # 텍스트→이미지 방향
    li = -(m * logp_i2t).sum(1) / m.sum(1).clamp(min=1)
    lt = -(m * logp_t2i).sum(0) / m.sum(0).clamp(min=1)
    return (li.mean() + lt.mean()) / 2


def supcon_loss(feats, labels, temperature=0.1):
    """같은 design_id(=label)를 positive로 보는 supervised contrastive loss.
    feats: [B, D] L2 정규화 이미지 임베딩. 뷰 불변 표현 학습용."""
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.t() / temperature
    B = feats.size(0)
    self_mask = torch.eye(B, dtype=torch.bool, device=feats.device)
    sim.masked_fill_(self_mask, float("-inf"))

    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    valid = pos_mask.any(dim=1)                       # positive가 있는 앵커만
    if valid.sum() == 0:
        return feats.new_tensor(0.0)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
    return -mean_log_prob[valid].mean()


@torch.no_grad()
def _encode_for_hobit(model, processor, imgs, device, dtype):
    """HOBIT 배치 구성용 이미지 임베딩 [B, D] (L2 정규화). 증강 없이 결정적으로."""
    px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        emb = model.get_image_features(pixel_values=px)
    if not torch.is_tensor(emb):          # MetaCLIP 2는 출력 객체를 반환
        emb = emb.pooler_output
    emb = F.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy().astype("float32")


def _pos_mask(design_label):
    """레코드 i,j가 같은 design_id면 positive. 대각선 포함, 대칭.

    제목 동일성은 positive 근거가 되지 못한다. 고유 제목 28,859개 중 유일한 것이
    141개뿐이라(2026-08 실측) 'Shoe'가 4,720건 반복된다. 제목이 같다고 묶으면 서로
    다른 디자인이 정답으로 취급돼, 특히 locarno_aware 배치에서 마스크가 거의 전부
    True가 되고 변별 그래디언트가 사라진다.
    """
    return design_label[:, None] == design_label[None, :]


def _recall(logits, pos):
    """멀티-positive 인식 R@{1,5,10}: top-K 안에 정답이 하나라도 있으면 히트."""
    ranks = logits.argsort(dim=1, descending=True)
    ps = pos.gather(1, ranks)                          # 정렬 순서로 정답 재배열
    return (ps[:, :1].any(1).sum().item(), ps[:, :5].any(1).sum().item(),
            ps[:, :10].any(1).sum().item(), logits.size(0))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """양방향 Recall@{1,5,10} (배치 단위 프록시). 공식 수치는 eval_retrieval.py
    (전체 갤러리 R@K/mAP) 기준 — 여기는 학습 중 빠른 추세 확인용."""
    model.eval()
    t = [0, 0, 0]; i = [0, 0, 0]; n = 0
    for bi, enc in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        d = enc.pop("design_label").to(device)
        pos = _pos_mask(d)                             # 대칭 → 양방향 공용
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"])
        *tr, B = _recall(out.logits_per_text, pos)     # Text→Image
        *ir, _ = _recall(out.logits_per_image, pos)    # Image→Text
        for k in range(3):
            t[k] += tr[k]; i[k] += ir[k]
        n += B
    model.train()
    return {"T2I_R@1": t[0]/n, "T2I_R@5": t[1]/n, "T2I_R@10": t[2]/n,
            "I2T_R@1": i[0]/n, "I2T_R@5": i[1]/n, "I2T_R@10": i[2]/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--image-root", default="data")
    ap.add_argument("--max-steps", type=int, default=0, help="스모크런: N 옵티마이저 스텝 후 종료")
    ap.add_argument("--limit", type=int, default=0, help="스모크런: 레코드 N개만 사용")
    ap.add_argument("--eval-batches", type=int, default=0, help="평가 배치 수 제한(0=전체)")
    ap.add_argument("--log-every", type=int, default=0, help="config 로그 주기 오버라이드")
    ap.add_argument("--num-workers", type=int, default=-1, help="config num_workers 오버라이드")
    ap.add_argument("--batch-size", type=int, default=0, help="config batch_size 오버라이드")
    ap.add_argument("--grad-accum", type=int, default=0, help="config grad_accum 오버라이드")
    ap.add_argument("--epochs", type=int, default=0, help="config epochs 오버라이드")
    ap.add_argument("--save-every", type=int, default=0, help="N 스텝마다 체크포인트 저장(긴 학습 안전장치)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    if args.log_every:
        t["log_every"] = args.log_every
    if args.num_workers >= 0:
        t["num_workers"] = args.num_workers
    if args.batch_size:
        t["batch_size"] = args.batch_size
    if args.grad_accum:
        t["grad_accum"] = args.grad_accum
    if args.epochs:
        t["epochs"] = args.epochs
    set_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[t["precision"]]

    model, processor = build_model(cfg)
    model.to(device)

    records = load_records(t["data_path"])
    records = take_limit(records, args.limit, t["seed"])
    # design_id 단위 분할: 같은 디자인의 뷰가 train/eval 양쪽에 갈리는 누수 방지
    train_recs, eval_recs = split_by_design(records, t["eval_ratio"], t["seed"])
    print(f"split by design_id: train {len(train_recs)} / eval {len(eval_recs)} records")

    collate_train = Collator(processor, cfg["model"]["image_size"], args.image_root,
                             augment=t.get("augment", True))
    collate_eval = Collator(processor, cfg["model"]["image_size"], args.image_root)

    sampler = None
    which = t.get("sampler", "pk" if t.get("pk_views", 4) > 1 else "random")
    if which == "hobit":
        sampler = HobitBatchSampler(
            train_recs, t["batch_size"], pool=t.get("hobit_pool", 4096),
            penalty=t.get("hobit_penalty", 10.0),
            mask_false_negatives=t.get("mask_false_negatives", True), seed=t["seed"])
    elif which == "pk":
        sampler = PKBatchSampler(train_recs, t["batch_size"],
                                 views_per_design=t.get("pk_views", 4),
                                 locarno_aware=t.get("locarno_aware", True), seed=t["seed"])
    if sampler is not None:
        train_loader = DataLoader(PairDataset(train_recs), batch_sampler=sampler,
                                  num_workers=t["num_workers"], collate_fn=collate_train)
    else:                                  # 기존 랜덤 배치 (baseline)
        train_loader = DataLoader(PairDataset(train_recs), batch_size=t["batch_size"],
                                  shuffle=True, num_workers=t["num_workers"],
                                  collate_fn=collate_train, drop_last=True)
    eval_loader = DataLoader(PairDataset(eval_recs), batch_size=t["batch_size"],
                             shuffle=False, num_workers=t["num_workers"], collate_fn=collate_eval)

    # 파라미터 그룹: LoRA는 높은 LR, logit_scale은 낮은 LR
    logit_scale = model.base_model.model.logit_scale
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "logit_scale" not in n]
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": t["lr_lora"], "weight_decay": t["weight_decay"]},
        {"params": [logit_scale], "lr": t["lr_logit_scale"], "weight_decay": 0.0},
    ])

    steps_per_epoch = math.ceil(len(train_loader) / t["grad_accum"])
    # max_steps 지정 시 스케줄(워밍업·코사인)도 그 총량 기준 → LR이 제대로 오르내림
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * t["epochs"]
    warmup = int(total_steps * t["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: (
        s / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))))

    mask_fn = t.get("mask_false_negatives", True)
    eval_batches = args.eval_batches or None
    os.makedirs(t["output_dir"], exist_ok=True)
    print(f"baseline: {evaluate(model, eval_loader, device, eval_batches)}")

    step = 0
    stop = False
    for epoch in range(t["epochs"]):
        if isinstance(sampler, HobitBatchSampler) and \
                epoch % max(1, t.get("hobit_refresh_every", 1)) == 0:
            # 배치 구성이 "현재" 모델 기준이어야 hard negative가 의미를 갖는다.
            # 학습 집합 전체를 1회 추론하는 비용이 에폭마다 든다 → refresh_every로 조절.
            # refresh_embeddings가 eval/train 전환을 책임진다 — 인코딩 중 예외(OOM 등)가
            # 나도 model.train()이 복구되도록 finally로 감싸져 있다 (src/hobit.py).
            with torch.no_grad():
                emb = refresh_embeddings(
                    model, train_recs, args.image_root, cfg["model"]["image_size"],
                    lambda imgs: _encode_for_hobit(model, processor, imgs, device, dtype))
            sampler.set_embeddings(emb)
            print(f"[hobit] epoch {epoch}: 임베딩 {emb.shape} 갱신", flush=True)
        if stop:
            break
        for i, enc in enumerate(train_loader):
            design_label = enc.pop("design_label").to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            # 마스킹 비활성 시 대각선만 positive = 표준 CLIP InfoNCE와 동일
            pos = _pos_mask(design_label) if mask_fn \
                else torch.eye(design_label.size(0), dtype=torch.bool, device=device)

            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            pixel_values=enc["pixel_values"])
                loss = masked_clip_loss(out.logits_per_image, pos)   # 이미지↔텍스트
                if t["img2img_weight"] > 0:
                    loss = loss + t["img2img_weight"] * supcon_loss(out.image_embeds, design_label)
                loss = loss / t["grad_accum"]

            loss.backward()

            if (i + 1) % t["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                with torch.no_grad():
                    logit_scale.clamp_(max=math.log(100))    # 온도 폭주 방지
                step += 1
                if step % t["log_every"] == 0:
                    print(f"e{epoch} s{step}/{total_steps} loss={loss.item()*t['grad_accum']:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}")
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(os.path.join(t["output_dir"], "latest"))
                    print(f"  [checkpoint] step {step} -> latest")
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break

        metrics = evaluate(model, eval_loader, device, eval_batches)
        print(f"[epoch {epoch}] {metrics}")
        model.save_pretrained(os.path.join(t["output_dir"], f"epoch{epoch}"))  # LoRA 어댑터만 저장

    model.save_pretrained(os.path.join(t["output_dir"], "final"))
    print("saved LoRA adapter ->", t["output_dir"])


if __name__ == "__main__":
    main()
