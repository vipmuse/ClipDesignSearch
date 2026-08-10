"""부분집합 생성: design 단위 + 로카르노 분포 유지."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_subset import build_subset  # noqa: E402


def _recs():
    """디자인 300개 × 뷰 8장 = 2400 도면. 로카르노 3종을 60/30/10 비율로."""
    out = []
    for d in range(300):
        code = "0101" if d < 180 else ("0202" if d < 270 else "0303")
        for v in range(8):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}",
                        "locarno": code, "text": "t"})
    return out


def test_한_디자인의_뷰는_전부_포함되거나_전부_제외된다():
    got = build_subset(_recs(), target_drawings=800, seed=42)
    per = collections.Counter(r["design_id"] for r in got)
    assert set(per.values()) == {8}, f"뷰가 쪼개진 디자인이 있다: {per}"


def test_목표_도면수에_근접한다():
    got = build_subset(_recs(), target_drawings=800, seed=42)
    assert 720 <= len(got) <= 880, f"목표 800에서 벗어남: {len(got)}"


def test_로카르노_분포가_유지된다():
    recs = _recs()
    got = build_subset(recs, target_drawings=800, seed=42)
    def share(rs):
        c = collections.Counter(r["locarno"] for r in rs)
        return {k: v / len(rs) for k, v in c.items()}
    src, sub = share(recs), share(got)
    for code in src:
        # 0.05는 너무 헐거웠다: 로카르노를 무시하고 아무 design이나 뽑는 샘플러도
        # 500 seed 중 154번(약 31%)만 걸린다. 실제 구현의 최대 편차는 200 seed에서
        # 정확히 0.0이라 0.01로 조여도 비용이 없고, 계층화 없는 구현은 사실상 항상 실패한다.
        assert abs(src[code] - sub.get(code, 0)) < 0.01, f"{code} 비율이 크게 틀어짐"


def test_같은_seed는_같은_결과():
    a = build_subset(_recs(), 800, seed=42)
    b = build_subset(_recs(), 800, seed=42)
    assert [r["image"] for r in a] == [r["image"] for r in b]


def test_목표가_전체보다_크면_원본_그대로():
    recs = _recs()
    assert len(build_subset(recs, 999999, seed=42)) == len(recs)


def test_입력_순서를_유지한다():
    recs = _recs()
    got = build_subset(recs, 800, seed=42)
    keep = {r["image"] for r in got}
    expected = [r["image"] for r in recs if r["image"] in keep]
    assert [r["image"] for r in got] == expected
