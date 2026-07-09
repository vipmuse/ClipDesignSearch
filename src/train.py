"""대조학습 루프: CLIP 이미지↔텍스트 + (선택) 이미지↔이미지 supervised contrastive.

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

from dataset import Collator, PairDataset, load_records
from model import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


def _recall(logits):
    """대각선이 정답인 유사도 행렬에서 R@{1,5,10} 히트 수 반환."""
    B = logits.size(0)
    target = torch.arange(B, device=logits.device)
    ranks = logits.argsort(dim=1, descending=True)
    hit = (ranks == target[:, None])
    return hit[:, :1].any(1).sum().item(), hit[:, :5].any(1).sum().item(), hit[:, :10].any(1).sum().item(), B


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """양방향 Recall@{1,5,10} (배치 단위 근사). Text→Image, Image→Text 모두.
    도면→도면(I→I)은 전체 갤러리 필요 → embed.py의 FAISS 인덱스로 오프라인 평가."""
    model.eval()
    t = [0, 0, 0]; i = [0, 0, 0]; n = 0
    for bi, enc in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        enc.pop("design_label", None)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"])
        *tr, B = _recall(out.logits_per_text)        # Text→Image
        *ir, _ = _recall(out.logits_per_image)       # Image→Text
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
    random.shuffle(records)
    if args.limit:
        records = records[:args.limit]
    n_eval = max(1, int(len(records) * t["eval_ratio"]))
    eval_recs, train_recs = records[:n_eval], records[n_eval:]

    collate = Collator(processor, cfg["model"]["image_size"], args.image_root)
    train_loader = DataLoader(PairDataset(train_recs), batch_size=t["batch_size"],
                              shuffle=True, num_workers=t["num_workers"],
                              collate_fn=collate, drop_last=True)
    eval_loader = DataLoader(PairDataset(eval_recs), batch_size=t["batch_size"],
                             shuffle=False, num_workers=t["num_workers"], collate_fn=collate)

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

    eval_batches = args.eval_batches or None
    os.makedirs(t["output_dir"], exist_ok=True)
    print(f"baseline: {evaluate(model, eval_loader, device, eval_batches)}")

    step = 0
    stop = False
    for epoch in range(t["epochs"]):
        if stop:
            break
        for i, enc in enumerate(train_loader):
            design_label = enc.pop("design_label").to(device)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            pixel_values=enc["pixel_values"], return_loss=True)
                loss = out.loss                              # CLIP 이미지↔텍스트 InfoNCE
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
