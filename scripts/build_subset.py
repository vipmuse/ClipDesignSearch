"""로카르노 계층화 design 단위 부분집합 생성 (방법 비교 단계용).

전체 472k 도면으로 5개 방법을 학습하는 것은 비현실적이다(방법당 14,769스텝/epoch).
로카르노 코드 비율을 유지한 채 design_id 단위로 뽑아 축소 표본을 만든다.
design 단위인 이유: 한 디자인의 뷰가 쪼개지면 img2img supcon 신호가 깨지고
split_by_design 홀드아웃과도 어긋난다.

  python scripts/build_subset.py --target 100000 --out data/subset_100k.jsonl
"""
import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import load_records  # noqa: E402


def build_subset(records, target_drawings, seed):
    """로카르노 비율을 유지하며 design 단위로 target_drawings장 근처까지 뽑는다.

    반환은 입력 순서를 유지한 부분 리스트 (진단·재현이 쉽도록).

    코드마다 **최소 1개 디자인은 반드시 남는다**: got이 0에서 시작하고 quota는 항상
    0보다 크므로 첫 디자인을 넣기 전에는 break가 걸리지 않는다. 의도한 동작이다 —
    희귀 코드가 통째로 사라지면 그 코드는 학습·평가에서 아예 관측되지 않는다.
    대가는 작은 target에서의 초과다: 코드 271종을 쓰는 실데이터에서 --target 2000이면
    바닥값만 271개 디자인(≈2,200 도면)이라 목표를 넘고 희귀 코드가 과대표집된다.
    target이 (코드 수 × 디자인당 평균 뷰 수)보다 충분히 커야 비율이 지켜진다.
    """
    if target_drawings >= len(records):
        return records

    # design_id -> 도면 수, design_id -> 로카르노(첫 레코드 기준)
    per_design = collections.Counter()
    design_code = {}
    for r in records:
        d = r.get("design_id", r["image"])
        per_design[d] += 1
        design_code.setdefault(d, (r.get("locarno") or "").strip())

    by_code = collections.defaultdict(list)
    for d in per_design:
        by_code[design_code[d]].append(d)

    total = len(records)
    rs = random.Random(seed)
    keep = set()
    for code, designs in sorted(by_code.items()):        # 정렬로 결정성 확보
        rs.shuffle(designs)
        code_drawings = sum(per_design[d] for d in designs)
        quota = target_drawings * code_drawings / total   # 이 코드가 가져갈 도면 수
        got = 0
        for d in designs:
            if got >= quota:
                break
            keep.add(d)
            got += per_design[d]

    return [r for r in records if r.get("design_id", r["image"]) in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pairs.jsonl")
    ap.add_argument("--target", type=int, default=100000, help="목표 도면 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/subset_100k.jsonl")
    args = ap.parse_args()

    records = load_records(os.path.join(ROOT, args.data))
    sub = build_subset(records, args.target, args.seed)
    designs = len({r.get("design_id") for r in sub})
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"subset: {len(sub)} drawings / {designs} designs -> {out}")


if __name__ == "__main__":
    main()
