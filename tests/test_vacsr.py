"""VACSR 어댑터·손실 테스트. GPU·모델 로딩 없이 CPU로 돈다."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from vacsr import (VacsrAdapter, _ste_prob, vacsr_loss, pairwise_score,
                   save_adapter, load_adapter)


def _adapter(dim=16, seed=0):
    torch.manual_seed(seed)
    return VacsrAdapter(dim, hidden=32)


def test_STE는_극단_로짓에서도_그래디언트가_소실되지_않는다():
    """logit=8이면 sigmoid'≈3e-4라 순정 MSE의 그래디언트는 사실상 0이다.
    STE는 |grad| = |y - sigmoid(logit)|을 유지해야 한다. 순정 경로와의 비교가
    이 테스트의 변별력이다 - STE를 빼면 아래 첫 assert가 깨진다."""
    logit = torch.tensor([8.0], requires_grad=True)
    y = torch.tensor([0.0])                       # false negative 상황: 라벨 0, 예측 1
    loss = 0.5 * (y - _ste_prob(logit)).pow(2).sum()
    loss.backward()
    p = torch.sigmoid(torch.tensor(8.0))
    assert torch.allclose(logit.grad, (p - y), atol=1e-5), \
        f"STE 그래디언트 {logit.grad.item():.6f} != {p - y}"

    plain = torch.tensor([8.0], requires_grad=True)
    (0.5 * (y - torch.sigmoid(plain)).pow(2).sum()).backward()
    assert plain.grad.abs().item() < 1e-3, "순정 경로가 소실되지 않으면 픽스처가 무의미"


def test_손실이_유한하고_KL이_비음수다():
    a = _adapter()
    torch.manual_seed(1)
    v = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss, stats = vacsr_loss(a, v, t)
    assert torch.isfinite(loss)
    assert stats["kl"] >= 0
    loss.backward()                               # 모든 헤드로 그래디언트가 흐른다
    grads = [p.grad for p in a.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_분산의_최적값은_오차_제곱이다():
    """논문 Eq 9: dL_σ/dσ̂=0 ⇔ σ̂² = 오차². 그 점에서 그래디언트가 0인지 확인한다."""
    err2 = torch.tensor(0.09)                     # 오차 0.3
    sig2 = err2.clone().requires_grad_(True)      # σ̂² = 오차²에 놓는다
    l = err2 / (2 * sig2) + 0.5 * sig2.log()
    l.backward()
    assert abs(sig2.grad.item()) < 1e-6
    # 최적점에서 벗어나면 복원력이 있는 방향으로 그래디언트가 선다
    sig2b = torch.tensor(0.5, requires_grad=True)
    (err2 / (2 * sig2b) + 0.5 * sig2b.log()).backward()
    assert sig2b.grad.item() > 0                  # σ̂²가 크면 줄이는 방향


def test_평가_경로는_결정적이다():
    a = _adapter()
    torch.manual_seed(2)
    txt = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
    img = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
    s1 = pairwise_score(a, txt, img)
    torch.manual_seed(999)                        # 시드를 흔들어도 같아야 한다
    s2 = pairwise_score(a, txt, img)
    assert torch.equal(s1, s2)
    assert s1.shape == (5, 7) and (s1 >= 0).all() and (s1 <= 1).all()


def test_pairwise_청크가_전체_계산과_같다():
    a = _adapter()
    torch.manual_seed(3)
    txt = torch.nn.functional.normalize(torch.randn(9, 16), dim=-1)
    img = torch.nn.functional.normalize(torch.randn(11, 16), dim=-1)
    assert torch.allclose(pairwise_score(a, txt, img, chunk_q=2, chunk_n=3),
                          pairwise_score(a, txt, img, chunk_q=100, chunk_n=100), atol=1e-6)


def test_저장_복원_후_점수가_같다(tmp_path):
    a = _adapter()
    save_adapter(a, str(tmp_path))
    b = load_adapter(str(tmp_path))
    torch.manual_seed(4)
    txt = torch.randn(3, 16)
    img = torch.randn(4, 16)
    assert torch.allclose(pairwise_score(a, txt, img), pairwise_score(b, txt, img))
    assert load_adapter(str(tmp_path / "없는폴더")) is None


def test_학습이_대각선을_올리고_비대각을_내린다():
    """작은 배치를 수백 스텝 과적합시켜 손실이 실제로 정렬을 학습하는지 본다.
    이게 통과하지 않으면 손실 부호나 STE 방향이 뒤집힌 것이다."""
    torch.manual_seed(5)
    a = _adapter(dim=8)
    v = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    # 초기 잠재 분산이 1 근처라(sigmoid(0)의 odds) 샘플링 노이즈가 커서 수렴이
    # 느리게 시작한다 - 오차가 줄어야 분산이 줄고 그래야 신호가 선명해지는 구조
    # 자체가 이 방법의 기제다. 1500 스텝이면 gap이 0.7을 넘는 것을 실측했다.
    for _ in range(1500):
        loss, _ = vacsr_loss(a, v, t)
        opt.zero_grad(); loss.backward(); opt.step()
    score = pairwise_score(a, t, v)               # [4, 4] (텍스트 쿼리 × 이미지)
    diag = score.diagonal().mean()
    offd = score[~torch.eye(4, dtype=torch.bool)].mean()
    assert diag > offd + 0.2, f"대각 {diag:.3f} vs 비대각 {offd:.3f}"


def test_잠재_분산이_샘플링_노이즈를_지배한다():
    """σ̂이 별도 헤드면 이 관계가 깨진다 - logvar 헤드의 bias를 키우면 같은 입력의
    반복 샘플 로짓 분산이 커져야 한다. FN 흡수(불확실한 쌍의 그래디언트 감쇠)가
    실재하는 경로인지 확인하는 기제 테스트."""
    torch.manual_seed(7)
    a = _adapter(dim=8)
    s_vec = torch.randn(1, 8)

    def logit_std(bias_val, n=200):
        with torch.no_grad():
            a.head_logvar.bias.fill_(bias_val)
        outs = []
        for _ in range(n):
            logit, _, _ = a(s_vec, sample=True)
            outs.append(logit)
        return torch.stack(outs).std().item()

    low, high = logit_std(-6.0), logit_std(6.0)   # odds: sigmoid(-6)≈0.0025 vs ≈400
    assert high > low * 5, f"분산 저={low:.4f} 고={high:.4f} - 잠재 분산이 노이즈와 무관"
