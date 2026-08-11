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


def test_손실이_전체_유효_배치를_본다():
    """청크 안이 아니라 전체에서 네거티브가 나와야 이 방법이 의미가 있다.
    8개를 4청크로 쪼갠 손실은, 2개짜리 배치 4개를 따로 계산한 손실과 달라야 한다."""
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)
    with torch.no_grad():
        i, t = enc(xi, xt)
    full, _, _, _ = cached_grads(i, t, pos, T, design, text, head, ls, compose_loss)

    per_chunk = []
    for s in range(0, 8, 2):
        sl = slice(s, s + 2)
        with torch.no_grad():
            ic, tc = enc(xi[sl], xt[sl])
        v, _, _, _ = cached_grads(ic, tc, pos[sl, sl], T, design[sl], text[sl], head[sl],
                                  ls, compose_loss)
        per_chunk.append(v)
    assert abs(full - sum(per_chunk) / len(per_chunk)) > 1e-3, \
        "전체 손실이 청크별 평균과 같다 - 네거티브가 청크 안에 갇혀 있다"


def test_stats가_compose_loss의_것을_그대로_전달한다():
    xi, xt, design, text, head, pos = _fixture()
    enc = _ToyEncoder()
    ls = torch.tensor(2.6592, requires_grad=True)
    with torch.no_grad():
        i, t = enc(xi, xt)
    _, _, _, stats = cached_grads(i, t, pos, T, design, text, head, ls, compose_loss)
    assert set(stats) == {"tic", "n_eligible", "n_violating"}
