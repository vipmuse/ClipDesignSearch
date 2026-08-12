"""hobit2 (HOBIT 논문 재현) 샘플러 테스트. GPU·모델 로딩 없이 CPU 텐서로 돈다."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hobit2 import Hobit2BatchSampler, build_batches, _int_labels


def _recs(n, designs=None, texts=None):
    return [{"image": f"i{k}.png",
             "design_id": (designs or list(range(n)))[k],
             "text": (texts or [f"t{k}" for k in range(n)])[k]} for k in range(n)]


def _norm(x):
    return F.normalize(torch.tensor(x, dtype=torch.float32), dim=-1)


def test_커버리지_모든_배치가_서로소이고_크기가_맞다():
    n, b = 50, 8
    g = torch.Generator(); g.manual_seed(0)
    img = F.normalize(torch.randn(n, 16, generator=g), dim=-1)
    txt = F.normalize(torch.randn(n, 16, generator=g), dim=-1)
    labels = torch.arange(n)
    batches = build_batches(img, txt, labels, labels, b, tau=0.05, generator=g)
    assert len(batches) == n // b
    flat = [i for batch in batches for i in batch]
    assert len(flat) == len(set(flat)), "인덱스가 배치 간에 중복됐다"
    assert all(len(batch) == b for batch in batches)


def _near_positive_fixture():
    """시드 1개(s=1이 되게 b=4, seed_frac=0.25)와 후보 3개.

    시드 0의 이미지 v0 근처에 있는 near-positive 후보(A)와, v0에서 멀지만 시드
    텍스트에 hard한 후보(B), 그리고 무관한 후보(C)를 만든다. 논문 스코어(λ=1)는
    B를 먼저 골라야 하고, λ=0(모순 항 제거)이면 A가 이긴다 — cross만 보면 A가
    시드와 더 비슷하기 때문. 이 반전이 이 테스트의 변별력이다.
    """
    d = 8
    e = torch.eye(d)
    v0 = e[0]                                   # 시드 이미지
    t0 = F.normalize(e[0] * 0.8 + e[1] * 0.6, dim=-1)   # 시드 텍스트 (v0와 정렬)
    # A: near-positive — 이미지가 v0와 거의 같고 텍스트도 시드와 비슷 → cross 높음, intra 높음
    vA = F.normalize(v0 + 0.1 * e[2], dim=-1)
    tA = F.normalize(t0 + 0.1 * e[3], dim=-1)
    # B: hard-but-distant — 텍스트가 시드 이미지와 정렬(hard)하되 이미지는 v0와 직교
    vB = e[4]
    tB = F.normalize(e[0] * 0.7 + e[4] * 0.7, dim=-1)
    # C: 무관
    vC, tC = e[5], e[6]
    img = torch.stack([v0, vA, vB, vC])
    txt = torch.stack([t0, tA, tB, tC])
    return img, txt


@pytest.mark.parametrize("lam,expect_first", [(1.0, 2), (0.0, 1)])
def test_논문_스코어의_모순_항이_선택을_뒤집는다(lam, expect_first):
    """λ=1이면 near-positive(A=인덱스1)를 제치고 distant-hard(B=인덱스2)를 먼저
    고르고, λ=0이면 A를 먼저 고른다. λ=0에서도 B가 이기면 픽스처가 모순 축을
    변별하지 못하는 것이므로 테스트 자체가 실패해야 한다."""
    img, txt = _near_positive_fixture()
    labels = torch.arange(4)
    g = torch.Generator(); g.manual_seed(0)
    # b=2, seed_frac=0.5 → 시드 1개 + greedy 선택 1개. 시드가 0번이 되도록 seed 탐색.
    for attempt in range(50):
        g2 = torch.Generator(); g2.manual_seed(attempt)
        batches = build_batches(img, txt, labels, labels, 2, tau=0.05,
                                seed_frac=0.5, topk=3, lam=lam, generator=g2)
        first = batches[0]
        if first[0] == 0:                        # 시드가 레코드 0
            assert first[1] == expect_first, \
                f"λ={lam}: 시드 0 다음 선택이 {first[1]} (기대 {expect_first})"
            return
    pytest.fail("시드 0으로 시작하는 셔플을 찾지 못했다")


def test_이산_모순_페널티가_같은텍스트_다른디자인을_미룬다():
    """연속 항이 놓칠 수 있는 라벨 모순(텍스트 동일, design 다름)을 페널티가 미룬다."""
    d = 8
    e = torch.eye(d)
    img = torch.stack([e[0], F.normalize(e[1] + 0.5 * e[0], dim=-1),
                       F.normalize(e[2] + 0.5 * e[0], dim=-1)])
    txt = torch.stack([e[3], e[4], e[5]])       # 텍스트 벡터는 전부 직교(연속 항 침묵)
    design = torch.tensor([0, 1, 2])
    text_lab = torch.tensor([7, 7, 8])          # 레코드 0·1이 같은 제목, 다른 디자인
    for attempt in range(50):
        g = torch.Generator(); g.manual_seed(attempt)
        batches = build_batches(img, txt, design, text_lab, 2, tau=0.05,
                                seed_frac=0.5, topk=3, lam=1.0, penalty=10.0,
                                generator=g)
        first = batches[0]
        if first[0] == 0:
            assert first[1] == 2, "같은 제목(라벨 모순) 후보 1이 페널티에도 뽑혔다"
            return
    pytest.fail("시드 0으로 시작하는 셔플을 찾지 못했다")


def test_작은_온도에서도_수치가_유한하다():
    n, b = 40, 8
    g = torch.Generator(); g.manual_seed(1)
    img = F.normalize(torch.randn(n, 16, generator=g), dim=-1)
    txt = F.normalize(torch.randn(n, 16, generator=g), dim=-1)
    labels = torch.arange(n)
    batches = build_batches(img, txt, labels, labels, b, tau=0.01, generator=g)
    assert len(batches) == n // b               # 완주했으면 NaN/inf 없이 argmax가 동작한 것


def test_임베딩_없으면_랜덤_폴백이고_행수_검증이_동작한다():
    recs = _recs(20)
    sam = Hobit2BatchSampler(recs, 8, seed=0)
    batches = list(iter(sam))
    assert len(batches) == 2 and all(len(b) == 8 for b in batches)
    with pytest.raises(ValueError):
        sam.set_embeddings(np.zeros((5, 4), "float32"), np.zeros((20, 4), "float32"), 0.05)


def test_샘플러가_에폭마다_다른_배치를_낸다():
    recs = _recs(30)
    sam = Hobit2BatchSampler(recs, 8, seed=0)
    g = torch.Generator(); g.manual_seed(0)
    img = F.normalize(torch.randn(30, 8, generator=g), dim=-1).numpy()
    txt = F.normalize(torch.randn(30, 8, generator=g), dim=-1).numpy()
    sam.set_embeddings(img, txt, 0.05)
    e1, e2 = list(iter(sam)), list(iter(sam))
    assert e1 != e2
