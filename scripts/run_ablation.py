"""정확도 개선 방법별 개별 학습(ablation) + 성능 기여 비교 리포트.

각 실험군(arm)마다: config 생성 → train.py 학습 → eval_retrieval.py 전체 갤러리
평가 → outputs/ablation/summary.md 비교표 생성. 모든 arm이 같은 seed/eval_ratio를
쓰므로 동일한 design_id 홀드아웃에서 비교된다.

  python scripts/run_ablation.py                      # 기본 arm 전체 (학습 6회!)
  python scripts/run_ablation.py --arms baseline pkmask all
  python scripts/run_ablation.py --epochs 3           # 짧은 ablation (권장 시작점)
  python scripts/run_ablation.py --quick              # 스모크: 레코드 2000개 × 1 epoch
  python scripts/run_ablation.py --report-only        # 기존 결과로 표만 재생성

기본 arm (각 방법을 baseline 위에 단독 추가 → 개별 기여 측정):
  base        튜닝 전 베이스 모델 (평가만, 학습 없음)
  baseline    LoRA 학습하되 모든 개선 OFF (기존 파이프라인과 동일 조건)
  aug         + 도면 증강 (소회전·스케일·라인두께·viewpoint 텍스트)
  mask        + false-negative 마스킹 InfoNCE
  pkmask      + PK 샘플러 + 마스킹 (세트 — PK 단독은 false negative를 늘려 유해)
  pkmask-i2i  + PK + 마스킹 + img2img supcon (supcon은 PK 없이는 미발화)
  all         전부 ON (= configs/lora_clip.yaml 기본값)

선택 arm (--arms로 명시할 때만; LoRA 용량 계열은 'all' 레시피 위에 추가):
  pk-only     PK 샘플러만, 마스킹 없이 (유해 상호작용 실증용)
  all-mlp     all + LoRA target에 fc1,fc2 추가
  all-proj    all + visual/text projection 학습
  all-r32     all + LoRA rank 16→32 (alpha 64)
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL_DIR = os.path.join(ROOT, "outputs", "ablation")

# 모든 개선 OFF = 기존 파이프라인 조건 (랜덤 배치, 표준 InfoNCE, 증강 없음)
OFF = {"pk_views": 1, "locarno_aware": False, "mask_false_negatives": False,
       "augment": False, "img2img_weight": 0.0}
ALL_ON = {"pk_views": 4, "locarno_aware": True, "mask_false_negatives": True,
          "augment": True, "img2img_weight": 0.5}


def _arm(base=OFF, lora=None, **delta):
    over = {"train": {**base, **delta}}
    if lora:
        over["lora"] = lora
    return over


ARMS = {
    "baseline":   _arm(),
    "aug":        _arm(augment=True),
    "mask":       _arm(mask_false_negatives=True),
    "pkmask":     _arm(pk_views=4, locarno_aware=True, mask_false_negatives=True),
    "pkmask-i2i": _arm(pk_views=4, locarno_aware=True, mask_false_negatives=True,
                       img2img_weight=0.5),
    "all":        _arm(base=ALL_ON),
    # ── 선택 arm ──
    "pk-only":    _arm(pk_views=4, locarno_aware=True),
    "all-mlp":    _arm(base=ALL_ON, lora={"target_modules":
                       ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]}),
    "all-proj":   _arm(base=ALL_ON, lora={"train_projections": True}),
    "all-r32":    _arm(base=ALL_ON, lora={"r": 32, "alpha": 64}),
}
DEFAULT_ARMS = ["baseline", "aug", "mask", "pkmask", "pkmask-i2i", "all"]


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) \
            if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def run(cmd, log_path):
    """서브프로세스 실행: 콘솔 에코 + arm별 로그 파일 동시 기록."""
    print(">>", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            print("   " + line.rstrip(), flush=True)
            log.write(line)
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"failed (exit {p.returncode}): {' '.join(cmd)}")


def eval_json_path(name):
    fname = "base.json" if name == "base" else "final.json"
    return os.path.join(ABL_DIR, name, "eval", fname)


def train_arm(name, base_cfg, args):
    adir = os.path.join(ABL_DIR, name)
    os.makedirs(adir, exist_ok=True)
    adapter = os.path.join(adir, "final")
    done = os.path.exists(os.path.join(adapter, "adapter_config.json"))
    if done and not args.force:
        print(f"[{name}] adapter 존재 → 학습 스킵 (--force로 재학습)")
        return adapter

    cfg = deep_merge(base_cfg, ARMS[name])
    cfg["train"]["output_dir"] = os.path.relpath(adir, ROOT)
    cfg_path = os.path.join(adir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    cmd = [sys.executable, os.path.join("src", "train.py"), "--config", cfg_path]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.eval_batches:
        cmd += ["--eval-batches", str(args.eval_batches)]
    run(cmd, os.path.join(adir, "train.log"))
    return adapter


def eval_arm(name, adapter, args):
    out = eval_json_path(name)
    if os.path.exists(out) and not args.force:
        print(f"[{name}] 평가 결과 존재 → 스킵")
        return
    cmd = [sys.executable, os.path.join("src", "eval_retrieval.py"),
           "--adapter", adapter, "--out", os.path.dirname(out)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    run(cmd, os.path.join(ABL_DIR, name, "eval.log"))


def report(arm_names):
    rows, missing = [], []
    for name in arm_names:
        p = eval_json_path(name)
        if os.path.exists(p):
            rows.append((name, json.load(open(p, encoding="utf-8"))))
        else:
            missing.append(name)
    if not rows:
        print("평가 결과가 없습니다."); return
    ref = dict(rows).get("baseline")

    def fmt(j, key, sub, delta=False):
        v = j.get(key, {}).get(sub)
        if v is None:
            return "-"
        if delta and ref and ref.get(key, {}).get(sub) is not None:
            return f"{v:.4f} ({v - ref[key][sub]:+.4f})"
        return f"{v:.4f}"

    header = ["arm", "I2I R@1", "I2I R@10", "I2I mAP", "T2I R@1", "T2I R@10", "T2I mAP"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for name, j in rows:
        d = name not in ("base", "baseline")     # baseline 대비 Δ 표기
        lines.append("| " + " | ".join([
            name,
            fmt(j, "I2I", "R@1", d), fmt(j, "I2I", "R@10", d), fmt(j, "I2I", "mAP", d),
            fmt(j, "T2I", "R@1", d), fmt(j, "T2I", "R@10", d), fmt(j, "T2I", "mAP", d),
        ]) + " |")
    if missing:
        lines.append(f"\n미완료 arm: {', '.join(missing)}")
    table = "\n".join(lines)

    os.makedirs(ABL_DIR, exist_ok=True)
    with open(os.path.join(ABL_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Ablation 결과 (Δ = baseline 대비)\n\n" + table + "\n")
    print("\n" + table)
    print(f"\nsaved -> {os.path.join(ABL_DIR, 'summary.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                    help=f"실험군 선택 (기본: {' '.join(DEFAULT_ARMS)})")
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--epochs", type=int, default=0, help="arm당 학습 epoch (짧은 ablation은 3~5 권장)")
    ap.add_argument("--limit", type=int, default=0, help="레코드 N개만 (train/eval 동일 적용)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="스모크: --limit 2000 --epochs 1")
    ap.add_argument("--force", action="store_true", help="기존 어댑터/평가 무시하고 재실행")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.limit = args.limit or 2000
        args.epochs = args.epochs or 1

    names = ["base"] + [a for a in args.arms if a != "base"]
    unknown = [a for a in names if a != "base" and a not in ARMS]
    if unknown:
        sys.exit(f"알 수 없는 arm: {unknown}  (가능: {list(ARMS)})")

    if args.report_only:
        report(names); return

    base_cfg = yaml.safe_load(open(os.path.join(ROOT, args.config), encoding="utf-8"))
    for name in names:
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        os.makedirs(os.path.join(ABL_DIR, name), exist_ok=True)
        if name == "base":
            eval_arm(name, "none", args)         # 튜닝 전 베이스라인: 평가만
        else:
            adapter = train_arm(name, base_cfg, args)
            eval_arm(name, adapter, args)
    report(names)


if __name__ == "__main__":
    main()
