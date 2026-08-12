"""VACSR: 변분 어댑터로 크로스모달 유사도를 연속 분포로 표현 (papers/(65621)2681).

이진 pos/neg 주석은 유사도 공간을 이진 경계로 압축해 false negative를 만든다 - 이
저장소가 세 번 실측한 병증(제목 다대다 중복, hobit 계열의 T2I 붕괴, tic 1차 설계
반증)과 정확히 같은 문제의식이다. VACSR은 유사도 자체를 잠재 변수로 두고 이진
라벨을 "노이즈 낀 관측"으로 취급한다: 라벨과 먼 쌍(=false negative 후보)에는 큰
분산이 배정되어 잘못된 그래디언트가 자동 감쇠된다 (논문 Eq 9: 최적 분산 = 오차²).

논문과의 대응:
- 유사도 벡터 s_ij = v_i ⊙ t_j (Hadamard, 차원 보존이 인코딩의 전제)
- 인코더가 2-성분 가우시안 혼합의 (μ, logσ²)와 혼합 가중치, 분산 헤드 σ̂을 예측
- Gumbel-Softmax로 성분 선택, 재매개변수화로 z 샘플, 디코더 μ(z)가 유사도 로짓
- 손실 = α·L_KL + L_recon + γ·(L_σ^P + L_σ^N), α=0.0005, γ=1 (논문 값)
- L_recon의 sigmoid 미분은 STE로 우회 (Appendix B Eq 13) - 유사도가 0/1 근처일 때
  그래디언트가 소실되는 것을 막는다
- 하드 네거티브(행별 최대 항)는 ŷ=1로 둔다 (Eq 10) - 유사도가 큰 네거티브에 높은
  불확실성을 주면 변별 학습 자체가 무너지기 때문

구현 결정 (논문이 명시하지 않은 지점, 재현 시 확인할 것):
- σ̂² 매개변수화: 논문은 "logσ²에 sigmoid를 적용해 치역을 보존하며 정의역을
  [0,∞)로 확장"이라고만 쓴다. t=sigmoid(raw)를 odds t/(1-t)로 (0,∞)에 사상한다.
- σ̂은 별도 헤드가 아니라 잠재 분산의 혼합 평균이다. 3.2절 제목("Uncertainty in
  Latent Variables")과 Eq 7의 전개(z = μ̂+ε·σ̂ 샘플링)가 근거 - 별도 헤드로 두면
  L_σ가 재구성 그래디언트에 아무 영향을 주지 못해 FN 흡수 기제가 죽는다.
- 추론은 결정적으로: 성분 가중 평균 μ를 z로 쓴다 (샘플링 없음). 평가가 실행마다
  달라지면 arm 비교가 성립하지 않는다.
- 혼합 가중치 ω는 쌍별로 인코더가 예측 (Gumbel 선택 분포와 일치시킨다).
- 손실은 fp32로 계산한다 - KL·logvar가 bf16에서 불안정하다.

추론 시 T2I 유사도 = sigmoid(μ(z)). 코사인이 아니므로 FAISS IP 인덱스로 직접 검색할
수 없다 - 평가는 쌍별 계산(pairwise_score), 서빙은 코사인 top-K 후 재랭킹이 후속
과제다. I2I는 크로스모달 어댑터의 적용 대상이 아니라 코사인을 유지한다.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_FILE = "vacsr_adapter.pt"


class VacsrAdapter(nn.Module):
    """유사도 벡터 [*, d] → (유사도 로짓, KL, σ̂). 2-성분 가우시안 혼합 VAE."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.dim = dim
        self.hidden = hidden
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.head_mu = nn.Linear(hidden, 2 * dim)        # 성분 2개의 μ
        self.head_logvar = nn.Linear(hidden, 2 * dim)    # 성분 2개의 logσ²
        self.head_mix = nn.Linear(hidden, 2)             # 혼합 로짓
        self.decoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, 1))

    def encode(self, s):
        h = self.trunk(s)
        mu = self.head_mu(h).unflatten(-1, (2, self.dim))          # [*, 2, d]
        raw = self.head_logvar(h).unflatten(-1, (2, self.dim))
        # 논문의 정의역 확장: logσ² 출력에 sigmoid를 걸어 (0,1), odds로 (0,∞)에 사상.
        # 이 var가 샘플링 노이즈이자 L_σ의 감독 대상이다 - 별도 헤드가 아니라 잠재
        # 분산 그 자체여야 FN 쌍의 그래디언트가 노이즈로 감쇠되는 기제가 성립한다
        # (3.2절 제목이 "Uncertainty in Latent Variables"인 이유).
        t = torch.sigmoid(raw)
        var = (t / (1 - t + 1e-6)).clamp(1e-6, 1e6)                # [*, 2, d]
        mix_logits = self.head_mix(h)                              # [*, 2]
        return mu, var, mix_logits

    def forward(self, s, sample=True, gumbel_tau=1.0):
        """반환: (유사도 로짓 μ(z) [*], KL [*], σ̂² [*] = 잠재 분산의 혼합 평균)."""
        mu, var, mix_logits = self.encode(s)
        omega = F.softmax(mix_logits, dim=-1)                      # [*, 2]
        # 성분별 KL(N(μ,σ²)||N(0,1))을 ω로 가중 (논문 Eq 5)
        kl_c = 0.5 * (mu.pow(2) + var - var.log() - 1).sum(-1)     # [*, 2]
        kl = (omega * kl_c).sum(-1)
        # 쌍별 불확실성 스칼라: ω-가중 성분 분산의 차원 평균. L_σ가 이것을 오차²로
        # 끌면 같은 파라미터가 만드는 샘플링 노이즈가 커져 해당 쌍의 재구성
        # 그래디언트가 평균적으로 흐려진다 - FN 흡수의 실제 경로.
        sig2 = (omega.unsqueeze(-1) * var).sum(-2).mean(-1)        # [*]
        if sample:
            w = F.gumbel_softmax(mix_logits, tau=gumbel_tau, hard=True)  # [*, 2]
            eps = torch.randn_like(mu)
            z_c = mu + eps * var.sqrt()                            # [*, 2, d]
            z = (w.unsqueeze(-1) * z_c).sum(-2)
        else:                       # 결정적 평가 경로: 혼합 평균
            z = (omega.unsqueeze(-1) * mu).sum(-2)
        logit = self.decoder(z).squeeze(-1)
        return logit, kl, sig2


def _ste_prob(logit):
    """sigmoid의 STE (Appendix B): 값은 sigmoid(logit), 그래디언트는 항등.

    MSE의 d/dlogit이 sigmoid'(≈0/1 근처에서 소실)을 곱하지 않게 해, 이진 라벨
    회귀가 극단값에서 멈추는 것을 막는다 - sigmoid loss의 그래디언트와 동치가 된다.
    """
    return torch.sigmoid(logit).detach() + logit - logit.detach()


def vacsr_loss(adapter, image_embeds, text_embeds, alpha=0.0005, gamma=1.0):
    """배치 [B,d] 쌍에서 VACSR 목적함수 (논문 Eq 11). 반환 (loss, stats).

    ŷ는 배치 대각선(i==j)만 1 - baseline InfoNCE와 같은 감독이다. 같은 design의
    다른 뷰나 같은 제목의 다른 디자인에 추가 라벨을 주지 않는 것이 의도다:
    false negative를 라벨 없이 흡수한다는 것이 이 방법의 주장이기 때문이다.
    """
    v = image_embeds.float()
    t = text_embeds.float()
    B = v.shape[0]
    s = v.unsqueeze(1) * t.unsqueeze(0)                            # [B, B, d]
    logit, kl, sig2 = adapter(s.reshape(B * B, -1), sample=True)
    logit, kl, sig2 = logit.view(B, B), kl.view(B, B), sig2.view(B, B)

    y = torch.eye(B, device=v.device)
    p = _ste_prob(logit)
    recon = 0.5 * (y - p).pow(2).mean()                            # Eq 6 (σ²=1)

    # L_σ (Eq 8-10): 오차는 가중치로만 쓴다(stop-grad) - 분산 헤드만 최적화.
    err2 = (y - torch.sigmoid(logit)).pow(2).detach()
    l_sig_pos = (err2.diagonal() / (2 * sig2.diagonal())
                 + 0.5 * sig2.diagonal().log()).mean()
    # 하드 네거티브: ŷ=1로 두고 행별 최대 항 (유사도 큰 네거티브가 변별을 이끈다)
    err2_hn = (1 - torch.sigmoid(logit)).pow(2).detach()
    off = ~torch.eye(B, dtype=torch.bool, device=v.device)
    term = err2_hn / (2 * sig2) + 0.5 * sig2.log()
    l_sig_neg = term.masked_fill(~off, float("-inf")).max(dim=1).values.mean()

    loss = alpha * kl.mean() + recon + gamma * (l_sig_pos + l_sig_neg)
    stats = {"recon": float(recon.detach()), "kl": float(kl.mean().detach()),
             "sig_pos": float(sig2.diagonal().mean().detach().sqrt()),
             "p_diag": float(torch.sigmoid(logit).diagonal().mean().detach())}
    return loss, stats


@torch.no_grad()
def pairwise_score(adapter, txt_emb, img_emb, chunk_q=16, chunk_n=2048):
    """텍스트 [Q,d] × 이미지 [N,d] → 유사도 [Q,N] (결정적). 평가·재랭킹용.

    쿼리·갤러리 양쪽을 청크로 나눈다 - [Q, N, d] 텐서를 통째로 만들면 갤러리
    1만 × 1024차원에서 수 GB가 된다. 16 × 2048 × 1024 = 약 134MB로 유지.
    """
    adapter.eval()
    img = img_emb.float()
    rows = []
    for qs in range(0, txt_emb.shape[0], chunk_q):
        tq = txt_emb[qs:qs + chunk_q].float()                      # [q, d]
        cols = []
        for ns in range(0, img.shape[0], chunk_n):
            iv = img[ns:ns + chunk_n]                              # [n, d]
            s = tq.unsqueeze(1) * iv.unsqueeze(0)                  # [q, n, d]
            q, n, d = s.shape
            logit, _, _ = adapter(s.reshape(q * n, d), sample=False)
            cols.append(torch.sigmoid(logit).view(q, n))
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows)


def save_adapter(adapter, ckpt_dir):
    torch.save({"dim": adapter.dim, "hidden": adapter.hidden,
                "state": adapter.state_dict()},
               os.path.join(ckpt_dir, ADAPTER_FILE))


def load_adapter(ckpt_dir, device="cpu"):
    """체크포인트 디렉터리에서 어댑터 복원. 없으면 None (cosine 평가로 폴백)."""
    p = os.path.join(ckpt_dir, ADAPTER_FILE)
    if not os.path.exists(p):
        return None
    blob = torch.load(p, map_location=device, weights_only=True)
    a = VacsrAdapter(blob["dim"], hidden=blob.get("hidden")).to(device)
    a.load_state_dict(blob["state"])
    a.eval()
    return a
