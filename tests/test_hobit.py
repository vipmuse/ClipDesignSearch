"""HOBIT 배치 샘플러: 커버리지·결정성·hardness·모순 회피를 고정."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hobit import HobitBatchSampler  # noqa: E402


def _recs(n_design=40, views=4, titles=("Shoe", "Bottle")):
    """디자인 n_design개 × 뷰 views장. 제목은 titles를 순환 → 다른 디자인이 같은 제목을 공유."""
    out = []
    for d in range(n_design):
        for v in range(views):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}",
                        "text": titles[d % len(titles)]})
    return out


def _two_clusters(n, dim=8, seed=0):
    """앞 절반과 뒤 절반이 서로 먼 두 군집. hardness가 작동하면 배치가 한 군집으로 뭉친다."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, dim)) * 0.05
    e[: n // 2, 0] += 1.0
    e[n // 2:, 0] -= 1.0
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype("float32")


def test_에폭당_인덱스가_중복없이_나온다():
    recs = _recs()                                  # 160 레코드
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    flat = [i for b in s for i in b]
    assert len(flat) == len(set(flat)), "같은 인덱스가 두 번 나왔다"
    assert len(flat) == (len(recs) // 16) * 16, "자투리 규약과 다르다"
    assert set(flat) <= set(range(len(recs)))


def test_모든_배치가_정확히_batch_size():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert all(len(b) == 16 for b in s)


def test_같은_seed_같은_에폭이면_같은_배치():
    recs = _recs()
    e = _two_clusters(len(recs))
    a = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); a.set_embeddings(e)
    b = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); b.set_embeddings(e)
    assert list(a) == list(b)


def test_에폭이_바뀌면_배치도_바뀐다():
    recs = _recs()
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7)
    s.set_embeddings(e)
    assert list(s) != list(s)                       # __iter__가 epoch을 증가시킨다


def test_임베딩이_없으면_랜덤_폴백하되_커버리지는_유지():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=3)
    flat = [i for b in s for i in b]                # set_embeddings 호출 없음
    assert len(flat) == len(set(flat)) == (len(recs) // 16) * 16


def test_hardness_배치가_한_군집으로_뭉친다():
    """무작위 배치라면 배치 내 평균 유사도가 0 근처. greedy가 작동하면 크게 양수."""
    recs = _recs(n_design=64, views=4)              # 256 레코드
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=0.0, seed=5)
    s.set_embeddings(e)
    sims = []
    for b in s:
        v = e[b]
        g = v @ v.T
        sims.append((g.sum() - np.trace(g)) / (len(b) * (len(b) - 1)))
    assert np.mean(sims) > 0.5, f"배치 내 유사도가 낮다 — greedy가 작동하지 않음: {np.mean(sims)}"


def test_마스킹_off면_같은_design_id를_같은_배치에_피한다():
    """mask_false_negatives=False면 같은 디자인의 뷰도 네거티브로 밀리므로 모순이다."""
    recs = _recs(n_design=64, views=4)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    dup = 0
    for b in s:
        ds = [recs[i]["design_id"] for i in b]
        dup += len(ds) - len(set(ds))
    assert dup == 0, f"같은 design_id가 한 배치에 {dup}번 겹쳤다"


def test_마스킹_on이면_같은_design_id는_허용하고_동일제목_타디자인만_피한다():
    recs = _recs(n_design=64, views=4)              # 제목 2종이 32개 디자인씩 공유
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    cross = 0
    for b in s:
        for x in range(len(b)):
            for y in range(x + 1, len(b)):
                rx, ry = recs[b[x]], recs[b[y]]
                if rx["text"] == ry["text"] and rx["design_id"] != ry["design_id"]:
                    cross += 1
    # 제목이 2종뿐이라 완전 회피는 불가능하지만, 무작위(배치16에서 약 60쌍)보다 훨씬 적어야 한다
    n_batches = len(recs) // 16
    assert cross / n_batches < 40, f"동일제목·타디자인 쌍이 배치당 {cross / n_batches:.1f}쌍"


def test_len은_배치_수와_일치():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert len(s) == len(list(s))
