"""정확도 개선 방법별 개별 학습(ablation) + 성능 기여 비교 리포트.

각 실험군(arm)마다: config 생성 → train.py 학습 → eval_retrieval.py 전체 갤러리
평가 → outputs/methods/summary.md 비교표 생성. 모든 arm이 같은 seed/eval_ratio를
쓰므로 동일한 design_id 홀드아웃에서 비교된다.

방법 정의는 configs/methods/<name>.yaml 에 있다 (src/registry.py가 병합의 유일한
진실원천). 새 arm을 추가하려면 YAML 한 장만 만들면 된다.

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
  hobit       baseline + submodular greedy 배치 구성 (에폭마다 학습셋 전체 재인코딩 → 비쌈)
  tic         baseline + 텍스트 모달 내부 대조 손실 (제목 중복 대응, 손실 축만 변경)
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
sys.path.insert(0, os.path.join(ROOT, "src"))
import registry  # noqa: E402

ABL_DIR = registry.OUT_ROOT
DEFAULT_ARMS = ["baseline", "aug", "mask", "pkmask", "pkmask-i2i", "all"]


def run(cmd, log_path):
    """서브프로세스 실행: 콘솔 에코 + arm별 로그 파일 동시 기록.

    한국어 로그가 자식에서 cp949로 나가면 부모의 utf-8 디코딩이 깨진다 — 자식의
    stdout 인코딩을 PYTHONIOENCODING으로 강제해 소스에서부터 utf-8이 되게 한다.
    """
    print(">>", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            log.write(line)                       # 로그 파일은 utf-8 — 원본 그대로 먼저 보존
            enc = sys.stdout.encoding or "utf-8"
            safe = line.rstrip().encode(enc, errors="replace").decode(enc, errors="replace")
            print("   " + safe, flush=True)        # 콘솔 코드페이지가 못 그리는 문자로 죽지 않게
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"failed (exit {p.returncode}): {' '.join(cmd)}")


def eval_json_path(name):
    fname = "base.json" if name == "base" else "final.json"
    return os.path.join(registry.method_dir(name), "eval", fname)


def resolved_path(name):
    return os.path.join(registry.method_dir(name), "config.resolved.yaml")


def diff_keys(saved, cur, prefix=""):
    """두 config에서 값이 다른 키를 'a.b.c' 형태로 나열. (키, 저장본, 현재) 리스트."""
    out = []
    for k in sorted(set(saved) | set(cur)):
        a, b = saved.get(k, "<없음>"), cur.get(k, "<없음>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff_keys(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append((prefix + k, a, b))
    return out


def is_trained(name, args):
    """arm의 어댑터가 이미 학습되어 있는지 (--force면 무조건 False).

    train_arm과 main()의 base 분기가 반드시 이 함수 하나로 판단해야 한다. 두 곳이
    따로 조건을 적으면 어긋나기 쉽다 — 예를 들어 --force로 baseline을 재학습하는
    문서화된 복구 경로에서, base가 먼저 실행되며 아직 갱신 전인 옛 저장본을 읽고
    baseline은 새로 학습하면, base와 baseline이 서로 다른 홀드아웃/설정을 보게
    되어 비교표의 기준점(base 행)이 조용히 틀어진다. 학습 도중 중단된 경우도
    같다 — config.resolved.yaml은 학습 시작 전에 쓰이므로, 어댑터 없이 config만
    남을 수 있다(이때도 False가 나와야 한다).
    """
    adapter = os.path.join(registry.method_dir(name), "final")
    return os.path.exists(os.path.join(adapter, "adapter_config.json")) and not args.force


def train_arm(name, args):
    """resolved config와 데이터 포인터를 못박고 학습. (어댑터 경로, config 경로) 반환.

    학습을 건너뛸 때는 저장된 config.resolved.yaml을 절대 덮어쓰지 않는다. 덮어쓰면
    재현 기록이어야 할 파일이 어댑터와 무관하게 변하고, 이후 평가·인덱스가 학습되지
    않은 설정(예: 224로 학습한 어댑터를 384로)으로 돌면서 index_meta.json까지 그
    거짓말에 동의해버린다.
    """
    adir = registry.method_dir(name)
    adapter = os.path.join(adir, "final")
    cfg = registry.resolve(name)
    cfg_path = resolved_path(name)
    trained = is_trained(name, args)

    if trained and os.path.exists(cfg_path):
        stored = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
        diffs = diff_keys(stored, cfg)
        if diffs:
            print(f"!! [{name}] 저장된 config.resolved.yaml과 현재 메서드 YAML이 다릅니다 "
                  f"— 저장본을 그대로 쓰고 진행합니다 (어댑터는 저장본 설정으로 학습됨).")
            for key, was, now in diffs:
                print(f"!!   {key}: 저장본={was!r} 현재={now!r}")
            print(f"!! [{name}] 새 설정으로 학습·평가·인덱스를 다시 만들려면 --force 로 재실행하세요.")
        else:
            print(f"[{name}] adapter 존재 → 학습 스킵 (--force로 재학습)")
        return adapter, cfg_path

    cfg_path = registry.write_resolved(name, cfg)
    registry.write_data_pointer(name, cfg)          # 산출물에 데이터 출처를 남긴다
    if trained:
        print(f"[{name}] adapter 존재 → 학습 스킵 (--force로 재학습)")
        return adapter, cfg_path

    cmd = [sys.executable, os.path.join("src", "train.py"), "--config", cfg_path]
    for flag, val in (("--epochs", args.epochs), ("--limit", args.limit),
                      ("--max-steps", args.max_steps), ("--eval-batches", args.eval_batches)):
        if val:
            cmd += [flag, str(val)]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    run(cmd, os.path.join(adir, "train.log"))
    return adapter, cfg_path


def eval_arm(name, adapter, cfg_path, args):
    out = eval_json_path(name)
    if os.path.exists(out) and not args.force:
        print(f"[{name}] 평가 결과 존재 → 스킵")
        return
    cmd = [sys.executable, os.path.join("src", "eval_retrieval.py"),
           "--adapter", adapter, "--out", os.path.dirname(out),
           "--config", cfg_path]                      # 학습과 동일 홀드아웃 보장
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    run(cmd, os.path.join(registry.method_dir(name), "eval.log"))


def index_arm(name, adapter, cfg_path, args):
    """방법별 FAISS 인덱스 구축. 웹 비교(Phase 4~5)가 이 산출물을 쓴다.

    embed.py build의 --data 기본값(data/pairs.jsonl)은 config와 무관한 하드코딩이라,
    데이터를 명시하지 않으면 train/eval이 실제로 쓴 train.data_path와 다른 파일을
    인덱싱할 수 있다(예: 어떤 방법이 subset_100k.jsonl을 쓰면). resolved config에서
    직접 읽어 넘겨 세 단계가 항상 같은 데이터를 보게 한다.
    """
    idx_dir = os.path.join(registry.method_dir(name), "index")
    if os.path.exists(os.path.join(idx_dir, "index_meta.json")) and not args.force:
        print(f"[{name}] 인덱스 존재 → 스킵")
        return
    data_path = yaml.safe_load(open(cfg_path, encoding="utf-8"))["train"]["data_path"]
    cmd = [sys.executable, os.path.join("src", "embed.py"), "build",
           "--config", cfg_path, "--adapter", adapter, "--index", idx_dir,
           "--data", data_path]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    run(cmd, os.path.join(registry.method_dir(name), "index.log"))


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
    ap.add_argument("--epochs", type=int, default=0, help="arm당 학습 epoch (짧은 ablation은 3~5 권장)")
    ap.add_argument("--limit", type=int, default=0, help="레코드 N개만 (train/eval 동일 적용)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=0)
    ap.add_argument("--image-root", default="", help="이미지 루트 (train/eval 동일 적용)")
    ap.add_argument("--quick", action="store_true", help="스모크: --limit 2000 --epochs 1")
    ap.add_argument("--force", action="store_true", help="기존 어댑터/평가 무시하고 재실행")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-index", action="store_true", help="인덱스 빌드 생략")
    args = ap.parse_args()
    if args.quick:
        args.limit = args.limit or 2000
        args.epochs = args.epochs or 1

    names = ["base"] + [a for a in args.arms if a != "base"]
    known = registry.list_methods()
    unknown = [a for a in names if a != "base" and a not in known]
    if unknown:
        sys.exit(f"알 수 없는 메서드: {unknown}  (가능: {known})")

    if args.report_only:
        report(names); return

    for name in names:
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        os.makedirs(registry.method_dir(name), exist_ok=True)
        if name == "base":
            # 튜닝 전 베이스 모델: 학습 없이 평가만. baseline의 resolved를 기준 config로 쓴다.
            # "이미 학습됨" 판정은 train_arm과 반드시 같은 predicate(is_trained)를 써야
            # 한다 — base는 names 목록 맨 앞이라 baseline보다 먼저 실행되므로, 조건이
            # 어긋나면(--force 재실행, 혹은 학습 중단으로 config만 남고 어댑터가 없는
            # 경우) base와 baseline이 서로 다른 config.resolved.yaml을 보게 되어
            # 비교표의 기준점이 조용히 틀어진다.
            _baseline_cfg_path = resolved_path("baseline")
            base_cfg = (_baseline_cfg_path
                        if is_trained("baseline", args) and os.path.exists(_baseline_cfg_path)
                        else registry.write_resolved("baseline"))
            eval_arm(name, "none", base_cfg, args)
        else:
            adapter, cfg_path = train_arm(name, args)
            eval_arm(name, adapter, cfg_path, args)
            if not args.no_index:
                index_arm(name, adapter, cfg_path, args)
    report(names)


if __name__ == "__main__":
    main()
