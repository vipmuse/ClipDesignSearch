"""GradCache: 물리 배치와 유효 네거티브 수를 분리한다.

대조학습의 네거티브는 물리 배치 안에서 나오므로 grad_accum으로는 늘릴 수 없다.
활성값을 청크 하나 분량만 유지한 채 손실은 전체를 보게 하려면 두 번 순전파해야 한다:
1패스는 no_grad로 임베딩만 모으고, 그 위에서 손실을 계산해 임베딩에 대한 그래디언트를
얻는다. 2패스는 청크별로 다시 순전파하며 그 그래디언트를 주입해 역전파한다.

모델이 logits를 (image_embeds @ text_embeds.T) * logit_scale.exp()로 계산하고
logit_bias가 없으므로, 캐시된 임베딩만으로 logits를 정확히 재구성할 수 있다.
이 구현이 근사가 아니라 등가인 근거다 (tests/test_gradcache.py가 고정한다).
"""
from types import SimpleNamespace

import torch


def cached_grads(image_embeds, text_embeds, pos, t, design_label, text_label, head_label,
                 logit_scale, compose_fn):
    """캐시된 임베딩으로 손실을 계산하고 임베딩에 대한 그래디언트를 돌려준다.

    반환: (손실값 float, grad_img, grad_txt, stats)

    logit_scale은 여기서 그래디언트를 받는다 - 2패스에서 다시 계산하지 않으므로
    이 backward가 logit_scale의 유일한 갱신 경로다.
    """
    img = image_embeds.detach().requires_grad_(True)
    txt = text_embeds.detach().requires_grad_(True)
    logits_per_image = (img @ txt.t()) * logit_scale.exp()
    out = SimpleNamespace(logits_per_image=logits_per_image, image_embeds=img, text_embeds=txt)
    loss, stats = compose_fn(out, pos, t, design_label, text_label, head_label)
    loss.backward()
    return float(loss.detach()), img.grad, txt.grad, stats
