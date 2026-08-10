"""TIC 손실: 무엇을 밀고 무엇을 빼는지 고정한다.

핵심은 필터다 — 같은 제목 쌍은 임베딩이 같은 벡터라 분리 불가능하고,
같은 design_id 쌍은 붙어 있는 것이 맞다. 둘 다 손실에서 빠져야 한다.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import tic_loss  # noqa: E402


def _emb(vectors):
    """행 단위 L2 정규화된 [B, D] 텐서."""
    t = torch.tensor(vectors, dtype=torch.float32)
    return t / t.norm(dim=-1, keepdim=True)


def test_제목이_같으면_밀어내지_않는다():
    """같은 문자열 → 같은 임베딩. 분리가 불가능하므로 손실 0이어야 한다."""
    e = _emb([[1.0, 0.0], [1.0, 0.0]])              # 유사도 1.0
    design = torch.tensor([0, 1])                    # 서로 다른 디자인
    text = torch.tensor([0, 0])                      # 같은 제목
    assert tic_loss(e, design, text, margin=0.5).item() == 0.0


def test_같은_design_id면_밀어내지_않는다():
    e = _emb([[1.0, 0.0], [0.99, 0.14]])
    design = torch.tensor([3, 3])                    # 같은 디자인
    text = torch.tensor([0, 1])                      # 다른 제목
    assert tic_loss(e, design, text, margin=0.5).item() == 0.0


def test_다른_디자인_다른_제목이_가까우면_양수_손실():
    e = _emb([[1.0, 0.0], [1.0, 0.02]])              # 유사도 ≈ 1.0
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    assert tic_loss(e, design, text, margin=0.5).item() > 0.0


def test_margin_이하로_떨어져_있으면_손실_0():
    e = _emb([[1.0, 0.0], [0.0, 1.0]])               # 유사도 0.0
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    assert tic_loss(e, design, text, margin=0.5).item() == 0.0


def test_가까울수록_손실이_커진다():
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    near = tic_loss(_emb([[1.0, 0.0], [1.0, 0.05]]), design, text, margin=0.5)
    far = tic_loss(_emb([[1.0, 0.0], [1.0, 0.60]]), design, text, margin=0.5)
    assert near.item() > far.item() > 0.0


def test_대상_쌍이_하나도_없으면_0을_반환하고_NaN이_아니다():
    e = _emb([[1.0, 0.0], [0.0, 1.0]])
    design = torch.tensor([5, 5])                    # 전부 같은 디자인 → 대상 없음
    text = torch.tensor([0, 0])
    out = tic_loss(e, design, text, margin=0.5)
    assert out.item() == 0.0 and torch.isfinite(out)


def test_그래디언트가_텍스트_임베딩으로_흐른다():
    e = _emb([[1.0, 0.0], [1.0, 0.05]]).requires_grad_(True)
    loss = tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), margin=0.5)
    loss.backward()
    assert e.grad is not None and e.grad.abs().sum().item() > 0


def test_배치가_커도_대칭_쌍을_두_번_세지_않는다():
    """상삼각만 세는지 확인 — 같은 쌍을 두 번 세면 평균이 왜곡되지는 않지만
    쌍 수 기반 통계를 나중에 붙일 때 어긋난다."""
    e = _emb([[1.0, 0.0], [1.0, 0.05], [1.0, 0.10]])
    design = torch.tensor([0, 1, 2])
    text = torch.tensor([0, 1, 2])
    out = tic_loss(e, design, text, margin=0.5)
    assert torch.isfinite(out) and out.item() > 0
