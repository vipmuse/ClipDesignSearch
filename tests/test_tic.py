"""TIC 손실: 무엇을 밀고 무엇을 빼는지 고정한다.

선택 규칙은 두 축이다 — 같은 헤드명사(물품군)이고 코사인이 상한 아래인 쌍만
floor 위로 밀린다. 그 위에 같은 제목 쌍(분리 불가능한 동일 벡터)과 같은
design_id 쌍(같은 디자인의 다른 뷰)을 필터에서 뺀다.
"""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import compose_loss, masked_clip_loss, tic_loss  # noqa: E402


def _emb(vectors):
    """행 단위 L2 정규화된 [B, D] 텐서."""
    t = torch.tensor(vectors, dtype=torch.float32)
    return t / t.norm(dim=-1, keepdim=True)


def test_제목이_같으면_밀어내지_않는다():
    """같은 문자열 → 같은 임베딩. 분리가 불가능하므로 손실 0이어야 한다."""
    e = _emb([[1.0, 0.0], [1.0, 0.0]])
    assert tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 0]),
                    floor=0.5).item() == 0.0


def test_같은_design_id면_밀어내지_않는다():
    """유사도 ≈0.8486 ([1,0]·[0.85,0.53] 정규화 후 코사인, 실측 확인함) — floor(0.5)와
    기본 ceiling(0.92) 사이라 design_label 축만으로 배제되는지 검증한다. 예전 벡터
    (0.99,0.14 → sim≈0.99)는 ceiling 위라 design 절을 지워도 통과해 축을 안 가렸다."""
    e = _emb([[1.0, 0.0], [0.85, 0.53]])
    design = torch.tensor([3, 3])                    # 같은 디자인
    text = torch.tensor([0, 1])                      # 다른 제목
    head = torch.tensor([0, 0])                       # 같은 물품군
    assert tic_loss(e, design, text, head, floor=0.5).item() == 0.0


def test_다른_디자인_다른_제목이_가까우면_양수_손실():
    e = _emb([[1.0, 0.0], [1.0, 0.5]])                # 유사도 ≈ 0.894 (floor<sim<ceiling)
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    head = torch.tensor([0, 0])                       # 같은 물품군이라야 대상이 된다
    assert tic_loss(e, design, text, head, floor=0.5).item() > 0.0


def test_floor_이하로_떨어져_있으면_손실_0():
    e = _emb([[1.0, 0.0], [0.0, 1.0]])               # 유사도 0.0
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    head = torch.tensor([0, 0])
    assert tic_loss(e, design, text, head, floor=0.5).item() == 0.0


def test_헤드명사가_다르면_밀어내지_않는다():
    """'Shoe'와 'Bottle'은 코사인이 높아도 서로 다른 물품군이라 대상이 아니다.
    1차 설계가 이 쌍을 밀어내려 한 것이 실패의 한 축이었다.

    유사도 ≈0.8486 ([1,0]·[0.85,0.53] 정규화 후 코사인, 실측 확인함) — floor(0.5)와
    기본 ceiling(0.92) 사이라 head_label 축만으로 배제되는지 검증한다. 예전 벡터
    (1.0,0.05 → sim≈0.999)는 ceiling 위라 head_label 절을 지워도 우연히 통과해,
    이 테스트가 이 태스크의 핵심 축(헤드명사)을 실제로 지키지 못하고 있었다."""
    e = _emb([[1.0, 0.0], [0.85, 0.53]])
    assert tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1]),
                    floor=0.5).item() == 0.0


def test_상한_이상이면_표기_차이로_보고_제외한다():
    """'Clothing hanger'/'Clothes hanger' 0.996 같은 쌍. 같은 물품이라 붙어 있어야 한다."""
    e = _emb([[1.0, 0.0], [1.0, 0.02]])              # 유사도 ≈ 0.9998
    out = tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]),
                   floor=0.5, ceiling=0.95)
    assert out.item() == 0.0


def test_같은_물품군이고_상한_아래면_밀어낸다():
    """'Pizza box'/'Storage box' 0.831 같은 목표 쌍."""
    e = _emb([[1.0, 0.0], [0.8, 0.6]])               # 유사도 0.8
    out = tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]),
                   floor=0.5, ceiling=0.95)
    assert out.item() > 0.0


def test_상한_경계_바로_아래는_대상이고_바로_위는_아니다():
    e = _emb([[1.0, 0.0], [0.9, 0.4359]])            # 유사도 ≈ 0.900
    d, tl, hl = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0])
    assert tic_loss(e, d, tl, hl, floor=0.5, ceiling=0.95).item() > 0.0
    assert tic_loss(e, d, tl, hl, floor=0.5, ceiling=0.85).item() == 0.0


def test_stats의_대상_쌍_수는_상한_적용_후_기준이다():
    """로그의 '대상 쌍 수'가 상한 전 숫자면 필터가 얼마나 걸러내는지 볼 수 없다."""
    e = _emb([[1.0, 0.0], [1.0, 0.02], [0.8, 0.6]])  # 0-1은 상한 위, 0-2/1-2는 아래
    d = torch.tensor([0, 1, 2]); tl = torch.tensor([0, 1, 2]); hl = torch.tensor([0, 0, 0])
    _, n_el, _ = tic_loss(e, d, tl, hl, floor=0.5, ceiling=0.95, return_stats=True)
    assert n_el == 2, f"상한 위 쌍이 대상에 남아 있다 (n_eligible={n_el})"


def test_가까울수록_손실이_커진다():
    design = torch.tensor([0, 1])
    text = torch.tensor([0, 1])
    head = torch.tensor([0, 0])
    # 둘 다 기본 ceiling(0.92) 아래 — near(sim≈0.857) far(sim≈0.707)
    near = tic_loss(_emb([[1.0, 0.0], [1.0, 0.6]]), design, text, head, floor=0.5)
    far = tic_loss(_emb([[1.0, 0.0], [1.0, 1.0]]), design, text, head, floor=0.5)
    assert near.item() > far.item() > 0.0


def test_대상_쌍이_하나도_없으면_0을_반환하고_NaN이_아니다():
    e = _emb([[1.0, 0.0], [0.0, 1.0]])
    design = torch.tensor([5, 5])                    # 전부 같은 디자인 → 대상 없음
    text = torch.tensor([0, 0])
    head = torch.tensor([0, 0])
    out = tic_loss(e, design, text, head, floor=0.5)
    assert out.item() == 0.0 and torch.isfinite(out)


def test_그래디언트가_텍스트_임베딩으로_흐른다():
    e = _emb([[1.0, 0.0], [1.0, 0.6]]).requires_grad_(True)   # 유사도 ≈ 0.857 (기본 ceiling 아래)
    loss = tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]), floor=0.5)
    loss.backward()
    assert e.grad is not None and e.grad.abs().sum().item() > 0


def test_배치가_커도_대칭_쌍을_두_번_세지_않는다():
    """상삼각만 세는지 확인.

    sim이 대칭이라 쌍을 두 번 세도 '평균'은 그대로다 — 손실값만 보는 검사는 상삼각
    마스크를 지워도 통과한다. 그래서 쌍 수를 직접 못박는다. 로그로 찍히는 위반/대상
    쌍 수가 2배로 부풀면 floor 스윕을 그 수치로 읽을 수 없다."""
    # 0°, 25°, 50° — 세 쌍 모두 유사도가 floor(0.5)와 기본 ceiling(0.92) 사이에 들도록.
    e = _emb([[1.0, 0.0], [0.9063, 0.4226], [0.6428, 0.7660]])
    design = torch.tensor([0, 1, 2])
    text = torch.tensor([0, 1, 2])
    head = torch.tensor([0, 0, 0])
    out, n_eligible, n_violating = tic_loss(e, design, text, head, floor=0.5, return_stats=True)
    assert torch.isfinite(out) and out.item() > 0
    assert n_eligible == 3, f"3개 레코드의 상삼각 쌍은 3개여야 한다 (받은 값 {n_eligible})"
    assert n_violating == 3


def test_손실값이_hinge_평균과_정확히_일치한다():
    """정규화를 못박는다. .mean()을 .sum()으로 바꾸면 배치 496쌍 기준 실효 강도가
    ~496배 뛰는데, 값을 재지 않는 검사는 전부 초록으로 통과한다."""
    e = _emb([[1.0, 0.0], [0.5, 3 ** 0.5 / 2], [0.0, 1.0]])   # 0°, 60°, 90°
    design = torch.tensor([0, 1, 2])
    text = torch.tensor([0, 1, 2])
    head = torch.tensor([0, 0, 0])
    # 유사도: (0,1)=cos60=0.5, (0,2)=cos90=0, (1,2)=cos30=√3/2 — 모두 기본 ceiling 아래
    # floor 0.5 → hinge [0, 0, √3/2-0.5], 상삼각 3쌍으로 나눈다
    expected = (3 ** 0.5 / 2 - 0.5) / 3
    loss, n_eligible, n_violating = tic_loss(e, design, text, head, floor=0.5, return_stats=True)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert (n_eligible, n_violating) == (3, 1)


def test_return_stats는_필터에_걸린_쌍을_대상에서_뺀다():
    """대상 쌍 수는 필터 뒤 값이어야 한다 — '값이 0'인 이유가 '밀 게 없어서'인지
    '필터가 다 걸러서'인지 로그에서 갈라야 처방이 갈린다."""
    e = _emb([[1.0, 0.0], [1.0, 0.05], [0.6, 0.8]])
    design = torch.tensor([0, 0, 1])          # (0,1)은 같은 디자인 → 제외
    text = torch.tensor([0, 1, 1])            # (1,2)는 같은 제목 → 제외
    head = torch.tensor([0, 0, 0])            # 같은 물품군 (제외 대상 아닌 (0,2)는 대상이어야 함)
    loss, n_eligible, n_violating = tic_loss(e, design, text, head, floor=0.5, return_stats=True)
    assert (n_eligible, n_violating) == (1, 1)   # (0,2)만 남는다 (유사도 0.6)
    assert loss.item() > 0

    # 대상이 하나도 없어도 통계 경로가 NaN/예외 없이 0을 준다
    zero, n_e, n_v = tic_loss(e, torch.tensor([5, 5, 5]), torch.tensor([0, 0, 0]), head,
                              floor=0.5, return_stats=True)
    assert zero.item() == 0.0 and (n_e, n_v) == (0, 0)


def test_return_stats_기본값은_기존_반환을_바꾸지_않는다():
    """기존 호출부(테스트 포함)가 텐서 하나를 그대로 받는지 고정."""
    e = _emb([[1.0, 0.0], [1.0, 0.05]])
    out = tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]), floor=0.5)
    assert torch.is_tensor(out) and out.dim() == 0


# ── compose_loss: 게이트가 실제로 켜지는지 (모델 없이) ──
# 브랜치의 값어치 전부가 학습 루프의 네 줄에 걸려 있는데, 그 네 줄은 지금까지 어떤
# 테스트도 지나가지 않았고 런타임 출력도 확인해주지 않았다. 게이트 오타 하나면
# tic arm이 이름만 tic인 baseline이 된다 — ablation의 최악 실패 모드다.

def _out(text_embeds):
    """모델 출력 스텁. compose_loss는 out의 세 필드만 읽으므로 모델도 GPU도 필요 없다."""
    B = text_embeds.size(0)
    return types.SimpleNamespace(
        logits_per_image=torch.eye(B) * 4.0,      # 대각선이 정답인 평범한 로짓
        image_embeds=torch.eye(B),
        text_embeds=text_embeds)


def _violating_batch():
    """서로 다른 디자인·다른 제목·같은 물품군이 floor 위로 붙어 있는 배치
    (TIC이 발화하는 조건). 유사도 (0,1)=0.8, (0,2)=0.0, (1,2)=0.6 — 셋 다 기본
    ceiling(0.92) 아래라 head_label을 같게 주면 그대로 대상이 된다."""
    e = _emb([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    head = torch.tensor([0, 0, 0])
    return _out(e), torch.eye(3, dtype=torch.bool), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), head


def test_tic_weight가_0이면_CLIP_단독과_정확히_같다():
    out, pos, design, text, head = _violating_batch()
    t = {"img2img_weight": 0.0, "tic_weight": 0.0, "tic_floor": 0.5}
    total, stats = compose_loss(out, pos, t, design, text, head)
    assert total.item() == masked_clip_loss(out.logits_per_image, pos).item()
    assert stats == {"tic": 0.0, "n_eligible": 0, "n_violating": 0}


def test_tic_weight가_켜지면_총_손실이_실제로_커진다():
    """게이트가 조용히 죽어 있으면 이 검사가 잡는다."""
    out, pos, design, text, head = _violating_batch()
    clip = masked_clip_loss(out.logits_per_image, pos).item()
    t = {"img2img_weight": 0.0, "tic_weight": 0.2, "tic_floor": 0.5}
    total, stats = compose_loss(out, pos, t, design, text, head)
    assert total.item() > clip
    assert stats["n_violating"] >= 1 and stats["tic"] > 0
    assert total.item() == pytest.approx(clip + 0.2 * stats["tic"], abs=1e-6)


def test_tic_기여가_tic_weight에_선형이다():
    """가중치를 두 배로 하면 기여도 정확히 두 배 — 스윕이 읽히려면 이게 성립해야 한다."""
    out, pos, design, text, head = _violating_batch()
    clip = masked_clip_loss(out.logits_per_image, pos).item()
    base = {"img2img_weight": 0.0, "tic_floor": 0.5}
    lo = compose_loss(out, pos, dict(base, tic_weight=0.1), design, text, head)[0].item() - clip
    hi = compose_loss(out, pos, dict(base, tic_weight=0.2), design, text, head)[0].item() - clip
    assert lo > 0
    assert hi == pytest.approx(2 * lo, rel=1e-5)


def test_floor가_커지면_TIC_기여가_줄어든다():
    """tic_floor 오타가 기본값(0.75)으로 조용히 떨어지는 사고를, 값이 실제로
    손실에 반영되는지 못박아 감지 가능하게 만든다."""
    out, pos, design, text, head = _violating_batch()
    base = {"img2img_weight": 0.0, "tic_weight": 0.2}
    tight = compose_loss(out, pos, dict(base, tic_floor=0.5), design, text, head)[1]["tic"]
    loose = compose_loss(out, pos, dict(base, tic_floor=0.99), design, text, head)[1]["tic"]
    assert tight > loose
