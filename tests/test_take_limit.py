"""--limit 축소 표본이 인제스천 순서에 편향되지 않는지 고정."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import take_limit  # noqa: E402


def _recs(n):
    # 앞 절반은 로카르노 0101, 뒤 절반은 0202 — 순서대로 자르면 한쪽만 뽑힌다
    return [{"image": f"{i}.png", "locarno": "0101" if i < n // 2 else "0202"}
            for i in range(n)]


def test_앞에서_자르지_않고_전_구간에서_뽑는다():
    got = take_limit(_recs(1000), 100, seed=42)
    codes = {r["locarno"] for r in got}
    assert codes == {"0101", "0202"}, "한쪽 구간만 뽑혔다 — 셔플이 빠졌다"
    assert len(got) == 100


def test_같은_seed는_같은_표본():
    a = take_limit(_recs(1000), 100, seed=42)
    b = take_limit(_recs(1000), 100, seed=42)
    assert [r["image"] for r in a] == [r["image"] for r in b]


def test_다른_seed는_다른_표본():
    a = take_limit(_recs(1000), 100, seed=1)
    b = take_limit(_recs(1000), 100, seed=2)
    assert [r["image"] for r in a] != [r["image"] for r in b]


def test_limit이_0이거나_전체보다_크면_원본_그대로():
    recs = _recs(50)
    assert take_limit(recs, 0, seed=42) is recs
    assert take_limit(recs, 100, seed=42) is recs
