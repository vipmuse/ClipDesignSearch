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
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import Collator, PairDataset, PKBatchSampler, load_records, split_by_design, take_limit
from gradcache import cached_grads
from hobit import HobitBatchSampler, refresh_embeddings
from model import build_model


# TIC 기본 파라미터. 리터럴이 세 곳(tic_loss 시그니처·compose_loss·main의 형변환)에
# 흩어져 있으면 어긋나도 조용하다 - compose_loss의 기본값이 손실에 들어가고 main의
# 기본값이 로그에 찍히므로, 갈라지면 로그가 손실이 쓰지도 않은 floor를 보여준다.
TIC_FLOOR = 0.75
TIC_CEILING = 0.92


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


def tic_loss(text_embeds, design_label, text_label, head_label,
             floor=TIC_FLOOR, ceiling=TIC_CEILING, return_stats=False):
    """텍스트 모달 내부 대조(TIC). 같은 물품군의 '서로 다른 물품'만 밀어낸다.

    선택 규칙은 두 축이다 (2026-08-11 실측으로 확정, 스펙 §3.2):
    - 같은 헤드명사: 'Pizza box'/'Storage box'처럼 관련 물품군으로 후보를 좁힌다.
      이 축이 없으면 'Shoe'/'Bottle'(코사인 0.867)까지 대상이 된다.
    - 코사인 < ceiling: 'Clothing hanger'/'Clothes hanger'(0.996)처럼 같은 물품의
      표기 차이를 뺀다. 이들은 붙어 있어야 맞다.
    여기에 문자열이 같은 쌍(같은 벡터라 분리 불가)과 같은 design_id 쌍, 그리고
    헤드명사가 비어 있는 쌍(head_label < 0)을 더 뺀다. 빈 헤드는 물품군의 근거가
    아니다 - 비-라틴 제목이 들어오면 전부 한 통에 묶여 헤드명사 축이 무효가 되고,
    TIC이 반증된 1차 설계(배치당 약 496쌍)로 조용히 되돌아간다.

    스칼라 임계값 하나로는 안 되는 이유: 베이스 모델 코사인은 물품 유사도 순으로
    정렬되지 않는다. 목표 쌍 Container/Beverage container는 0.786인데 무관한
    Shoe/Bottle이 0.867이고, 0.9를 넘는 것은 Eyeglasses/Glasses 같은 동의어다.
    물품군 선택과 표기차 배제는 서로 다른 축이라 하나로 겸할 수 없다.

    floor 위로 올라온 만큼만 hinge로 민다. floor 0.75는 같은 헤드명사 쌍 분포의
    중앙값이라, 배치당 대상 약 3쌍 중 절반쯤이 실제로 밀린다.

    return_stats=True면 (loss, 대상 쌍 수, 위반 쌍 수)를 돌려준다. 대상 쌍 수는
    상한을 적용한 뒤 기준이다 — 필터가 얼마나 걸러냈는지 로그에서 보이도록.
    """
    # .float(): bf16 autocast 아래서는 t @ t.t()도 bf16이라 임계 근처 표현 간격이
    # ~0.002다. F.normalize는 오늘의 MetaCLIP 2 출력에는 중복이지만 지우면 안 된다:
    # 이 손실을 행 단위 스케일 불변으로 만들어, 밀어내기가 노름을 부풀리거나
    # 무너뜨리는 방향으로 새지 않게 한다.
    t = F.normalize(text_embeds.float(), dim=-1)
    sim = t @ t.t()
    B = t.size(0)
    upper = torch.triu(torch.ones(B, B, dtype=torch.bool, device=t.device), diagonal=1)
    eligible = upper & (design_label[:, None] != design_label[None, :]) \
                     & (text_label[:, None] != text_label[None, :]) \
                     & (head_label[:, None] == head_label[None, :]) \
                     & (head_label[:, None] >= 0) \
                     & (sim < ceiling)
    if not eligible.any():
        zero = t.new_tensor(0.0)
        return (zero, 0, 0) if return_stats else zero
    hinge = (sim[eligible] - floor).clamp(min=0)
    loss = hinge.mean()
    # 통계는 파이썬 int로 낸다. 위 eligible.any()가 이미 매 스텝 GPU 동기화를
    # 유발하므로(if 조건이라 값이 필요하다) 카운트 두 개를 더 꺼내는 비용은 없다.
    return (loss, int(eligible.sum()), int((hinge > 0).sum())) if return_stats else loss


def compose_loss(out, pos, t, design_label, text_label, head_label):
    """총 손실 = CLIP + (img2img) + (TIC). out은 모델 출력 객체 또는 동일 필드를 가진 무엇이든.

    학습 루프에서 떼어낸 이유: 게이트가 실제로 켜지는지 모델 없이 검증하기 위해서다.
    tic_weight 오타나 잘못된 블록 배치로 tic arm이 조용히 baseline과 같아지는 사고가
    이 파이프라인의 최악 실패 모드다 — 며칠짜리 학습을 태우고 나서 summary.md의
    Δ≈0을 보고서야 드러난다.

    반환: (loss, stats). stats는 로그 전용 — tic 항의 가중 전 값과 대상/위반 쌍 수.
    게이트가 꺼져 있으면 전부 0이라 로그 포맷이 분기하지 않는다.
    """
    loss = masked_clip_loss(out.logits_per_image, pos)          # 이미지↔텍스트
    stats = {"tic": 0.0, "n_eligible": 0, "n_violating": 0}
    if t["img2img_weight"] > 0:
        loss = loss + t["img2img_weight"] * supcon_loss(out.image_embeds, design_label)
    if t.get("tic_weight", 0.0) > 0:
        tic, n_el, n_vi = tic_loss(out.text_embeds, design_label, text_label, head_label,
                                   floor=t.get("tic_floor", TIC_FLOOR),
                                   ceiling=t.get("tic_ceiling", TIC_CEILING), return_stats=True)
        loss = loss + t["tic_weight"] * tic
        # .detach(): requires_grad 텐서를 float()에 넣으면 로그 스텝마다 UserWarning이 뜬다.
        stats.update(tic=float(tic.detach()), n_eligible=n_el, n_violating=n_vi)
    return loss, stats


def _encode_batch(model, enc, device, dtype):
    """배치 하나를 순전파해 (out, 라벨들)을 돌려준다. 2패스 양쪽에서 쓴다."""
    design_label = enc.pop("design_label").to(device)
    text_label = enc.pop("text_label").to(device)
    head_label = enc.pop("head_label").to(device)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"])
    return out, design_label, text_label, head_label


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
        enc.pop("text_label", None)                    # 평가에는 쓰지 않는다
        enc.pop("head_label", None)                    # 평가에는 쓰지 않는다
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
    # 콘솔 코드페이지가 못 그리는 문자 하나가 학습 전체를 죽이지 않게 한다.
    # 러너(run_ablation.py)는 이미 같은 보호가 있지만 train.py를 직접 돌리면
    # 날 콘솔을 만난다 — 그 경로가 비어 있었다.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):        # 파이프로 리다이렉트된 경우 등
        pass

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
    if "tic_weight" in t:
        # YAML에서 따옴표가 붙으면(tic_weight: "0.2") 문자열이라 곱셈이 학습 도중
        # TypeError로 터진다. 몇 시간 뒤 첫 스텝이 아니라 지금 죽는 게 낫다.
        t["tic_weight"] = float(t["tic_weight"])
        t["tic_floor"] = float(t.get("tic_floor", TIC_FLOOR))
        t["tic_ceiling"] = float(t.get("tic_ceiling", TIC_CEILING))
    # YAML에서 따옴표가 붙으면 문자열이 들어온다. 형변환이 없으면 아래 비교가
    # TypeError로 죽고 원인이 설정에 있다는 게 드러나지 않는다 - tic_weight와 같은
    # 이유로 여기서 int로 고정한다.
    n_chunks = int(t.get("grad_cache_chunks", 1))
    if n_chunks > 1 and t["grad_accum"] > 1:
        # GradCache 경로는 grad_accum을 적용하지 않는다 - 조용히 한쪽을 무시하면
        # 유효 배치 크기가 설정과 달라져 config가 거짓말을 하게 된다.
        print("오류: grad_cache_chunks와 grad_accum을 동시에 1보다 크게 설정할 수 없습니다 "
              "(GradCache 경로는 grad_accum을 적용하지 않습니다).")
        sys.exit(1)
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

    if n_chunks > 1:
        # 청크 N개가 스텝 1회 - 자투리 그룹은 에폭 끝에서 버리므로 내림 나눗셈.
        # ceil을 쓰면 total_steps가 실제 옵티마이저 스텝 수보다 커져 코사인
        # 스케줄이 끝까지 내려오지 않는다.
        steps_per_epoch = len(train_loader) // n_chunks
        if steps_per_epoch == 0:
            # 배치 수가 n_chunks보다 적으면 그룹이 한 번도 안 채워져 옵티마이저
            # 스텝이 0회다. evaluate/save_pretrained는 그대로 돌고 종료 코드도
            # 0이라 "arm이 이름만 그 arm인 채로 완주"하는 조용한 무효화가 된다.
            print(f"오류: train_loader 배치 수({len(train_loader)})가 "
                  f"grad_cache_chunks({n_chunks})보다 적어 스텝이 0회가 됩니다. "
                  f"batch_size를 줄이거나 grad_cache_chunks를 줄이세요.")
            sys.exit(1)
    else:
        steps_per_epoch = math.ceil(len(train_loader) / t["grad_accum"])
    # max_steps 지정 시 스케줄(워밍업·코사인)도 그 총량 기준 → LR이 제대로 오르내림
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * t["epochs"]
    warmup = int(total_steps * t["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: (
        s / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))))

    mask_fn = t.get("mask_false_negatives", True)
    # 게이트가 켜졌다는 사실 자체를 로그에 남긴다 — tic arm이 조용히 baseline으로
    # 도는 사고는 학습 로그만 봐서는 baseline과 구분이 안 된다.
    tic_on = t.get("tic_weight", 0.0) > 0
    if tic_on:
        print(f"[tic] ON - tic_weight={t['tic_weight']} "
              f"floor={t['tic_floor']} ceiling={t['tic_ceiling']}", flush=True)
    eval_batches = args.eval_batches or None
    os.makedirs(t["output_dir"], exist_ok=True)
    print(f"baseline: {evaluate(model, eval_loader, device, eval_batches)}")

    step = 0
    stop = False
    for epoch in range(t["epochs"]):
        if stop:            # --max-steps로 끊긴 뒤 임베딩 갱신(실측 최대 ~2시간)을 낭비하지 않도록
            break           # 갱신 블록보다 반드시 위에 있어야 한다
        if isinstance(sampler, HobitBatchSampler) and \
                epoch % max(1, t.get("hobit_refresh_every", 1)) == 0:
            # 배치 구성이 "현재" 모델 기준이어야 hard negative가 의미를 갖는다.
            # 학습 집합 전체를 1회 추론하는 비용이 에폭마다 든다 → refresh_every로 조절.
            # 디코딩이 비용의 대부분(실측 78%)이라 학습 로더와 같은 num_workers로 병렬화한다.
            # refresh_embeddings가 eval/train 전환을 책임진다 — 인코딩 중 예외(OOM 등)가
            # 나도 model.train()이 복구되도록 finally로 감싸져 있다 (src/hobit.py).
            with torch.no_grad():
                emb = refresh_embeddings(
                    model, train_recs, args.image_root, cfg["model"]["image_size"],
                    lambda imgs: _encode_for_hobit(model, processor, imgs, device, dtype),
                    num_workers=t["num_workers"])
            sampler.set_embeddings(emb)
            print(f"[hobit] epoch {epoch}: 임베딩 {emb.shape} 갱신", flush=True)
        if n_chunks > 1:
            # GradCache 2패스 경로: 청크 N개를 모아 옵티마이저 스텝 1회를 만든다.
            # grad_accum은 여기서 적용하지 않는다 (시작 시점에 이미 상호배제 검증됨).
            group = []
            for enc in train_loader:
                group.append(enc)
                if len(group) < n_chunks:
                    continue                       # 자투리 그룹은 에폭 끝에서 버린다

                # 1패스: 임베딩만 모은다 (활성값은 청크 하나 분량만 유지).
                # 청크마다 RNG 상태를 저장한다 - lora.dropout이 0.05라 2패스가 다른
                # 드롭아웃 마스크를 보면 캐시한 그래디언트가 재계산된 순전파와 어긋난다.
                embeds, labels, rng_states = [], [], []
                with torch.no_grad():
                    for e in group:
                        rng_states.append((torch.get_rng_state(),
                                           torch.cuda.get_rng_state(device)
                                           if device == "cuda" else None))
                        out, d, tl, hl = _encode_batch(model, dict(e), device, dtype)
                        embeds.append((out.image_embeds.detach(), out.text_embeds.detach()))
                        labels.append((d, tl, hl))
                img = torch.cat([a for a, _ in embeds])
                txt = torch.cat([b for _, b in embeds])
                design = torch.cat([d for d, _, _ in labels])
                text_l = torch.cat([x for _, x, _ in labels])
                head_l = torch.cat([x for _, _, x in labels])

                # pos는 반드시 연결된 전체 라벨로 - 청크별로 만들면 네거티브가 갇힌다
                pos = _pos_mask(design) if mask_fn \
                    else torch.eye(design.size(0), dtype=torch.bool, device=device)
                # 기존 경로는 compose_loss를 autocast 안에서 부른다. 여기서도 같은
                # 컨텍스트로 감싸야 로짓 dtype이 맞는다 - 밖에서 부르면 로짓이 fp32가
                # 되어 bigbatch만 손실을 더 정밀하게 계산하고, Δ가 "네거티브 8배"가
                # 아니라 "네거티브 8배 + 정밀도"의 합이 된다(실측 그래디언트 차이 0.3~0.6%).
                with torch.autocast(device_type=device, dtype=dtype,
                                    enabled=(dtype != torch.float32)):
                    loss_val, g_img, g_txt, lstats = cached_grads(
                        img, txt, pos, t, design, text_l, head_l, logit_scale, compose_loss)

                # 2패스: 청크별 재순전파 + 그래디언트 주입.
                # 1패스에서 저장한 RNG 상태를 되돌려 같은 드롭아웃 마스크를 재현한다.
                off = 0
                for e, (cpu_state, cuda_state) in zip(group, rng_states):
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, device)
                    out, _, _, _ = _encode_batch(model, dict(e), device, dtype)
                    k = out.image_embeds.size(0)
                    torch.autograd.backward([out.image_embeds, out.text_embeds],
                                            [g_img[off:off + k], g_txt[off:off + k]])
                    off += k

                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                with torch.no_grad():
                    logit_scale.clamp_(max=math.log(100))    # 온도 폭주 방지
                step += 1
                group = []
                if step % t["log_every"] == 0:
                    msg = (f"e{epoch} s{step}/{total_steps} loss={loss_val:.4f} "
                           f"eff_batch={img.size(0)} "
                           f"lr={sched.get_last_lr()[0]:.2e}")
                    if tic_on:
                        # 이 로그가 왜 항 값과 위반/대상 쌍 수를 따로 찍는지는
                        # 아래 기존 경로의 같은 블록 주석 참조.
                        print(f"{msg} tic={lstats['tic']:.4e} "
                              f"(×{t['tic_weight']}→{t['tic_weight']*lstats['tic']:.4e}) "
                              f"위반/대상={lstats['n_violating']}/{lstats['n_eligible']}")
                    else:
                        print(msg)
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(os.path.join(t["output_dir"], "latest"))
                    print(f"  [checkpoint] step {step} -> latest")
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
        else:
            for i, enc in enumerate(train_loader):
                design_label = enc.pop("design_label").to(device)
                text_label = enc.pop("text_label").to(device)
                head_label = enc.pop("head_label").to(device)
                enc = {k: v.to(device) for k, v in enc.items()}
                # 마스킹 비활성 시 대각선만 positive = 표준 CLIP InfoNCE와 동일
                pos = _pos_mask(design_label) if mask_fn \
                    else torch.eye(design_label.size(0), dtype=torch.bool, device=device)

                with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                pixel_values=enc["pixel_values"])
                    # 키워드로 넘긴다. text_label과 head_label은 타입도 모양도 같아 자리가
                    # 바뀌어도 아무 데서도 안 터지고, 대신 eligible이 매 배치 공집합이 되어
                    # (제목이 같으면 헤드명사도 같다) tic arm이 이름만 tic인 baseline이 된다.
                    loss, lstats = compose_loss(out, pos, t, design_label=design_label,
                                                text_label=text_label, head_label=head_label)
                    loss = loss / t["grad_accum"]

                loss.backward()

                if (i + 1) % t["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                    with torch.no_grad():
                        logit_scale.clamp_(max=math.log(100))    # 온도 폭주 방지
                    step += 1
                    if step % t["log_every"] == 0:
                        msg = (f"e{epoch} s{step}/{total_steps} "
                               f"loss={loss.item()*t['grad_accum']:.4f} "
                               f"lr={sched.get_last_lr()[0]:.2e}")
                        if tic_on:
                            # 총 손실만 봐서는 "켜졌는데 대상이 없는지"와 "아예 안 켜졌는지"가
                            # 갈라지지 않는다 - 둘 다 총 손실이 baseline과 비슷해 보인다. 항의
                            # 값과 대상/위반 쌍 수를 따로 찍어야 그 둘이 구분되고, 가중 전 값을
                            # 찍으므로 총 손실 중 TIC 몫도 로그만으로 다시 계산된다.
                            print(f"{msg} tic={lstats['tic']:.4e} "
                                  f"(×{t['tic_weight']}→{t['tic_weight']*lstats['tic']:.4e}) "
                                  f"위반/대상={lstats['n_violating']}/{lstats['n_eligible']}")
                        else:
                            print(msg)
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
