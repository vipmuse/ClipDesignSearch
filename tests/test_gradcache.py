"""GradCache: 2패스가 단일 역전파와 같은 그래디언트를 내는지 고정한다.

이 방법의 존재 이유가 '메모리만 줄이고 수학은 그대로'이므로, 등가성이 깨지면
비교표의 Δ는 네거티브 수의 효과가 아니라 구현 오차가 된다.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gradcache import cached_grads  # noqa: E402
from train import compose_loss  # noqa: E402

T = {"img2img_weight": 0.0, "tic_weight": 0.0}          # 순수 CLIP 손실만


class _ToyEncoder(nn.Module):
    """이미지·텍스트를 각각 선형 변환 후 L2 정규화. 모델 대신 쓰는 최소 스탠드인."""

    def __init__(self, d_in=6, d_out=4):
        super().__init__()
        self.img = nn.Linear(d_in, d_out, bias=False)
        self.txt = nn.Linear(d_in, d_out, bias=False)

    def forward(self, xi, xt):
        i = torch.nn.functional.normalize(self.img(xi), dim=-1)
        t = torch.nn.functional.normalize(self.txt(xt), dim=-1)
        return i, t


def _fixture(n=8, d_in=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(n, d_in, generator=g)
    xt = torch.randn(n, d_in, generator=g)
    design = torch.arange(n)                             # 전부 다른 디자인
    text = torch.arange(n)
    head = torch.zeros(n, dtype=torch.long)
    pos = torch.eye(n, dtype=torch.bool)
    return xi, xt, design, text, head, pos


def _single_pass_grads(enc, logit_scale, xi, xt, pos, design, text, head):
    """비교 기준: 전체를 한 번에 순전파하고 한 번 역전파."""
    enc.zero_grad(set_to_none=True)
    if logit_scale.grad is not None:
        logit_scale.grad = None
    i, t = enc(xi, xt)
    logits = (i @ t.t()) * logit_scale.exp()
    from types import SimpleNamespace
    out = SimpleNamespace(logits_per_image=logits, image_embeds=i, text_embeds=t)
    loss, _ = compose_loss(out, pos, T, design, text, head)
    loss.backward()
    return float(loss), enc.img.weight.grad.clone(), enc.txt.weight.grad.clone(), \
        logit_scale.grad.clone()


def _gradcache_grads(enc, logit_scale, xi, xt, pos, design, text, head, n_chunks):
    """GradCache 2패스: 청크로 나눠 임베딩만 모으고, 손실 그래디언트를 주입해 역전파."""
    enc.zero_grad(set_to_none=True)
    if logit_scale.grad is not None:
        logit_scale.grad = None
    chunks = list(zip(xi.chunk(n_chunks), xt.chunk(n_chunks)))

    with torch.no_grad():                                # 1패스: 임베딩만
        embeds = [enc(a, b) for a, b in chunks]
    img = torch.cat([e[0] for e in embeds])
    txt = torch.cat([e[1] for e in embeds])

    _, g_img, g_txt, _ = cached_grads(img, txt, pos, T, design, text, head,
                                      logit_scale, compose_loss)

    off = 0                                              # 2패스: 재순전파 + 주입
    for a, b in chunks:
        i, t = enc(a, b)
        k = i.size(0)
        torch.autograd.backward([i, t], [g_img[off:off + k], g_txt[off:off + k]])
        off += k
    return enc.img.weight.grad.clone(), enc.txt.weight.grad.clone(), logit_scale.grad.clone()


def test_2패스_그래디언트가_단일_역전파와_같다():
    """이 방법의 존재 근거. 어긋나면 Δ가 구현 오차를 재게 된다."""
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)        # ln(100/7.2) 근처, 실제 초기값대
    base = {k: v.clone() for k, v in enc.state_dict().items()}

    _, gi_ref, gt_ref, gls_ref = _single_pass_grads(enc, ls, xi, xt, pos, design, text, head)
    enc.load_state_dict(base)
    gi, gt, gls = _gradcache_grads(enc, ls, xi, xt, pos, design, text, head, n_chunks=4)

    assert torch.allclose(gi, gi_ref, atol=1e-6), f"이미지 인코더 그래디언트 불일치 {(gi-gi_ref).abs().max()}"
    assert torch.allclose(gt, gt_ref, atol=1e-6), f"텍스트 인코더 그래디언트 불일치 {(gt-gt_ref).abs().max()}"
    assert torch.allclose(gls, gls_ref, atol=1e-6), "logit_scale 그래디언트 불일치"


def test_청크_수를_바꿔도_그래디언트가_같다():
    """청크 수는 메모리 손잡이일 뿐 수학을 바꾸지 않는다."""
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)
    base = {k: v.clone() for k, v in enc.state_dict().items()}
    outs = []
    for k in (1, 2, 4, 8):
        enc.load_state_dict(base)
        outs.append(_gradcache_grads(enc, ls, xi, xt, pos, design, text, head, n_chunks=k))
    for gi, gt, gls in outs[1:]:
        assert torch.allclose(gi, outs[0][0], atol=1e-6)
        assert torch.allclose(gt, outs[0][1], atol=1e-6)
        assert torch.allclose(gls, outs[0][2], atol=1e-6)


def _gradcache_grads_per_chunk(enc, logit_scale, xi, xt, pos, design, text, head, n_chunks):
    """일부러 틀리게 짠 대조군: 청크마다 cached_grads를 독립적으로 호출한다.

    이러면 각 청크의 손실이 그 청크 안의 원소만 네거티브로 본다 - GradCache가 막으려는
    바로 그 버그(1패스에서 concat한 전체 임베딩이 아니라 청크 단위로 loss를 매기는 것)를
    재현한 것이다. 이 헬퍼가 `_gradcache_grads`(정상 경로)와 다른 그래디언트를 내야
    "네거티브가 청크에 갇히지 않는다"는 주장이 메커니즘 수준에서 성립한다."""
    enc.zero_grad(set_to_none=True)
    if logit_scale.grad is not None:
        logit_scale.grad = None
    n = xi.size(0)
    chunk = n // n_chunks
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        with torch.no_grad():                            # 1패스: 이 청크만의 임베딩
            ic, tc = enc(xi[sl], xt[sl])
        # 손실이 이 청크 안의 pos/design/text/head만 본다 - 다른 청크는 네거티브 후보에서 빠진다.
        _, g_img, g_txt, _ = cached_grads(ic, tc, pos[sl, sl], T, design[sl], text[sl],
                                          head[sl], logit_scale, compose_loss)
        i, t = enc(xi[sl], xt[sl])                        # 2패스: 재순전파 + 주입
        torch.autograd.backward([i, t], [g_img, g_txt])
    return enc.img.weight.grad.clone(), enc.txt.weight.grad.clone(), logit_scale.grad.clone()


def test_네거티브가_청크에_갇히면_전체_배치_경로와_다른_그래디언트가_나온다():
    """실제 2패스 메커니즘(`_gradcache_grads`)을 청크별로 손실을 따로 매기는 대조군과
    비교한다. 두 경로가 같은 그래디언트를 낸다면 GradCache가 청크 경계를 넘어 네거티브를
    모으고 있다는 증거가 없는 것이다 - 이 테스트가 그 메커니즘을 직접 겨눈다."""
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)
    base = {k: v.clone() for k, v in enc.state_dict().items()}

    enc.load_state_dict(base)
    gi_full, gt_full, _ = _gradcache_grads(enc, ls, xi, xt, pos, design, text, head, n_chunks=4)

    enc.load_state_dict(base)
    ls2 = torch.tensor(2.6592, requires_grad=True)
    gi_trapped, gt_trapped, _ = _gradcache_grads_per_chunk(
        enc, ls2, xi, xt, pos, design, text, head, n_chunks=4)

    assert (gi_full - gi_trapped).abs().max() > 1e-3, \
        "청크에 갇힌 네거티브와 전체 유효 배치가 같은 이미지 그래디언트를 낸다"
    assert (gt_full - gt_trapped).abs().max() > 1e-3, \
        "청크에 갇힌 네거티브와 전체 유효 배치가 같은 텍스트 그래디언트를 낸다"


def _gradcache_grads_rng(enc, logit_scale, xi, xt, pos, design, text, head, n_chunks):
    """`_gradcache_grads`와 같되 RNG 상태를 저장/복원한다 (train.py의 실제 구현과 동일).

    1패스에서 청크마다 torch.get_rng_state()를 저장하고, 2패스 직전에
    torch.set_rng_state()로 되돌려 같은 드롭아웃 마스크를 재현한다."""
    enc.zero_grad(set_to_none=True)
    if logit_scale.grad is not None:
        logit_scale.grad = None
    chunks = list(zip(xi.chunk(n_chunks), xt.chunk(n_chunks)))

    embeds, rng_states = [], []
    with torch.no_grad():                                # 1패스: 임베딩만 + RNG 상태 저장
        for a, b in chunks:
            rng_states.append(torch.get_rng_state())
            embeds.append(enc(a, b))
    img = torch.cat([e[0] for e in embeds])
    txt = torch.cat([e[1] for e in embeds])

    _, g_img, g_txt, _ = cached_grads(img, txt, pos, T, design, text, head,
                                      logit_scale, compose_loss)

    off = 0                                              # 2패스: RNG 복원 + 재순전파 + 주입
    for (a, b), state in zip(chunks, rng_states):
        torch.set_rng_state(state)
        i, t = enc(a, b)
        k = i.size(0)
        torch.autograd.backward([i, t], [g_img[off:off + k], g_txt[off:off + k]])
        off += k
    return enc.img.weight.grad.clone(), enc.txt.weight.grad.clone(), logit_scale.grad.clone()


class _ToyEncoderDropout(nn.Module):
    """드롭아웃이 있는 스탠드인. 1패스와 2패스가 같은 마스크를 봐야 등가성이 성립한다."""

    def __init__(self, d_in=6, d_out=4, p=0.05):
        super().__init__()
        self.img = nn.Linear(d_in, d_out, bias=False)
        self.txt = nn.Linear(d_in, d_out, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, xi, xt):
        i = torch.nn.functional.normalize(self.drop(self.img(xi)), dim=-1)
        t = torch.nn.functional.normalize(self.drop(self.txt(xt)), dim=-1)
        return i, t


def _chunked_forward_grads(enc, logit_scale, xi, xt, pos, design, text, head, n_chunks):
    """드롭아웃 기준값: 청크 순서대로 순전파하되 그래프를 전부 들고 있다가 한 번만
    역전파한다 (GradCache가 절약하는 활성값 저장을 안 하는 것 빼고는 1패스와 동일한
    연산 순서).

    `_single_pass_grads`(전체를 한 번에 처리)를 기준으로 쓰면 안 된다: 전역 RNG는
    호출 순서에 의존하므로, 이미지 전체 배치를 한 번에 처리하는 것과 청크마다
    이미지→텍스트를 번갈아 처리하는 것은 같은 시드에서도 완전히 다른 드롭아웃 마스크를
    뽑는다(첫 청크 이후로는 경계 근처만이 아니라 전 구간이 어긋난다 - 실측 max diff
    0.86, 통계적으로 우연히 일치할 확률은 0.95**64 ≈ 3.6%뿐이다). GradCache가 지키는
    약속은 '청크로 나누지 않은 것과 같다'가 아니라 '청크 순서로 실제 그래프를 들고
    역전파한 것과 같다'이므로, 기준값도 같은 청크 순서로 RNG를 소비해야 한다."""
    enc.zero_grad(set_to_none=True)
    if logit_scale.grad is not None:
        logit_scale.grad = None
    chunks = list(zip(xi.chunk(n_chunks), xt.chunk(n_chunks)))
    outs = [enc(a, b) for a, b in chunks]                # 그래프 유지한 채 청크 순서로 순전파
    img = torch.cat([o[0] for o in outs])
    txt = torch.cat([o[1] for o in outs])
    logits = (img @ txt.t()) * logit_scale.exp()
    from types import SimpleNamespace
    out = SimpleNamespace(logits_per_image=logits, image_embeds=img, text_embeds=txt)
    loss, _ = compose_loss(out, pos, T, design, text, head)
    loss.backward()
    return enc.img.weight.grad.clone(), enc.txt.weight.grad.clone(), logit_scale.grad.clone()


def test_드롭아웃이_있어도_RNG를_되돌리면_등가성이_유지된다():
    """RNG 상태를 보존하지 않으면 2패스가 다른 마스크를 봐 그래디언트가 어긋난다.

    기준값은 `_chunked_forward_grads`(청크 순서 유지, 그래프 통짜 보관 후 1회 역전파) -
    이유는 위 docstring 참조. `torch.manual_seed(1234)`를 기준값과 GradCache 양쪽에
    동일하게 걸어, 1패스가 소비하는 RNG 시작점을 맞춘다."""
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoderDropout()
    enc.train()
    ls = torch.tensor(2.6592, requires_grad=True)
    base = {k: v.clone() for k, v in enc.state_dict().items()}

    torch.manual_seed(1234)
    gi_ref, gt_ref, _ = _chunked_forward_grads(enc, ls, xi, xt, pos, design, text, head, n_chunks=4)
    enc.load_state_dict(base)
    torch.manual_seed(1234)
    gi, gt, _ = _gradcache_grads_rng(enc, ls, xi, xt, pos, design, text, head, n_chunks=4)

    assert torch.allclose(gi, gi_ref, atol=1e-5), \
        f"드롭아웃 마스크가 재현되지 않았다 {(gi - gi_ref).abs().max()}"
    assert torch.allclose(gt, gt_ref, atol=1e-5)


def test_stats가_compose_loss의_것을_그대로_전달한다():
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)
    with torch.no_grad():
        i, t = enc(xi, xt)
    _, _, _, stats = cached_grads(i, t, pos, T, design, text, head, ls, compose_loss)
    assert set(stats) == {"tic", "n_eligible", "n_violating"}
