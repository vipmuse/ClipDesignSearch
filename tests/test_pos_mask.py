"""positive 마스크 회귀 테스트.

제목 동일성을 positive 근거로 삼으면 안 된다는 것을 고정한다. 실측(2026-08)상
고유 제목 28,859개 중 유일한 것이 141개뿐이라, 제목 기반 positive는 서로 다른
디자인을 정답으로 묶어 학습 신호를 없앤다.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import _pos_mask  # noqa: E402


def test_동일_제목_32개가_전부_다른_디자인이면_대각선만_positive():
    # 'Shoe' 32건이 한 배치에 모인 최악의 경우를 모사
    design = torch.arange(32)
    pos = _pos_mask(design)
    assert torch.equal(pos, torch.eye(32, dtype=torch.bool))
    assert pos.sum().item() == 32


def test_같은_design_id의_뷰끼리는_positive():
    design = torch.tensor([0, 0, 1, 1, 2])
    pos = _pos_mask(design)
    assert pos[0, 1] and pos[1, 0]
    assert not pos[0, 2]
    assert pos.sum().item() == 9          # 2x2 블록 두 개 + 단독 1


def test_대칭이고_대각선을_포함한다():
    design = torch.tensor([3, 3, 7])
    pos = _pos_mask(design)
    assert torch.equal(pos, pos.t())
    assert pos.diagonal().all()
