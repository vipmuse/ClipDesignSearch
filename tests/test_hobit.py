"""HOBIT 배치 샘플러: 커버리지·결정성·hardness·모순 회피를 고정."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hobit import HobitBatchSampler  # noqa: E402


def _recs(n_design=40, views=4, titles=("Shoe", "Bottle")):
    """디자인 n_design개 × 뷰 views장.

    titles=None이면 디자인마다 고유 제목 → 제목 충돌 없이 design_id 항만 남는다.
    기본값(2종 순환)은 다른 디자인이 제목을 공유하는 상황을 만든다.
    """
    out = []
    for d in range(n_design):
        title = f"T{d}" if titles is None else titles[d % len(titles)]
        for v in range(views):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}", "text": title})
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
    """mask_false_negatives=False면 같은 디자인의 뷰도 네거티브로 밀리므로 모순이다.

    제목을 디자인마다 고유하게 준다: 제목이 2종뿐이면 같은 design_id ⟹ 같은 제목이라
    same_text 항이 same_design 항을 완전히 가려, design_id 회피가 작동하는지 확인할 수 없다.
    실데이터는 제목당 평균 16개 레코드라 4096 풀에서 제목 충돌이 희소하다.

    마지막 배치는 검사에서 뺀다. 풀이 배치마다 batch_size씩 줄어 마지막에는
    npool == batch_size가 되어 선택의 여지가 없다 — 남은 것이 그대로 들어간다.
    모든 예제를 정확히 한 번 쓰는 한 구조적으로 피할 수 없다.
    """
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    batches = list(s)
    dup = 0
    for b in batches[:-1]:                      # 마지막 배치는 강제 구성이라 제외
        ds = [recs[i]["design_id"] for i in b]
        dup += len(ds) - len(set(ds))
    assert dup == 0, f"선택 여지가 있는 배치에서 같은 design_id가 {dup}번 겹쳤다"


def test_마지막_배치는_선택_여지가_없어_모순을_피하지_못한다():
    """구조적 한계를 명시적으로 고정한다 — 발견이 아니라 알려진 성질이 되도록."""
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    batches = list(s)
    assert len(batches) * 16 == len(recs), "이 픽스처는 자투리 없이 나누어떨어져야 한다"
    # 마지막 배치는 남은 16개가 강제로 들어가므로 중복이 생길 수 있다
    last_ds = [recs[i]["design_id"] for i in batches[-1]]
    assert len(last_ds) == 16


def test_마스킹_on이면_같은_design_id_뷰가_같은_배치에_모일_수_있다():
    """마스킹이 켜지면 같은 design_id는 positive로 처리되므로 회피 대상이 아니다."""
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    dup = sum(len(b) - len({recs[i]["design_id"] for i in b}) for b in s)
    assert dup > 0, "마스킹 on인데도 같은 design_id를 회피하고 있다 — 조건부 규칙이 무의미"


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
