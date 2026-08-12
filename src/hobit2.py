"""HOBIT 논문 재현 배치 샘플러 (papers/33267_HOBIT_..., ICML 2026 제출본).

기존 hobit.py는 전문 미확보 상태에서 초록의 원리만 차용했고, 스코어가 논문과
정반대였다: 논문은 w_ij = q_i·d_j − λ·d_i·d_j 로 "positive와 비슷한 네거티브"를
감점하는데(λ=1.0이 최적, 논문 Fig 4), 기존 구현은 이미지·이미지 유사도를 가점해
시각적 near-duplicate를 배치에 몰아넣었다. 논문 Fig 2가 약한 그래디언트의 원인으로
지목한 바로 그 구성이고, subset 실측의 T2I −2.9pp가 그 결과로 추정된다.

논문과의 대응 (Alg 2 기준):
- 시드 S: 배치의 seed_frac(기본 1/4). 작을수록 좋다는 민감도 실험을 따른다.
- 후보 풀 C: 시드당 cross-modal hardness 상위 topk(기본 200)의 합집합.
- 목적함수: LSE smooth-max (Eq 10). 온도 τ는 손실의 온도와 동일하게 — CLIP은
  logit_scale이 학습되므로 매 갱신 시점의 1/logit_scale.exp()를 받는다.
- greedy: marginal gain (Eq 16) 최대화. log-공간으로 계산해 τ~0.01에서도 안전하다.

논문은 단방향(query→document) 검색이고 CLIP은 양방향이라 w를 대칭화한다:
  w_ij = cross_ij − λ·intra_ij
  cross_ij = (t_i·v_j + v_i·t_j) / 2      # 각 방향의 hardness
  intra_ij = (v_i·v_j + t_i·t_j) / 2      # 각 방향의 모순 (positive와의 근접)
여기에 라벨 기반 이산 페널티를 더한다(다른 design_id인데 텍스트가 같은 쌍) —
논문은 라벨이 없어 연속 항으로만 처리하지만(RQ3), 우리는 라벨이 있으므로 기존
hobit과 같은 안전장치를 유지한다.
"""
import numpy as np
import torch


def _int_labels(values):
    """동일 값 → 동일 정수. (hobit.py와 동일한 계약)"""
    uniq = {}
    return np.array([uniq.setdefault(v, len(uniq)) for v in values], dtype=np.int64)


@torch.no_grad()
def build_batches(emb_img, emb_txt, design, text, batch_size, tau,
                  seed_frac=0.25, topk=200, lam=1.0, penalty=10.0,
                  mask_false_negatives=True, generator=None):
    """에폭 전체의 배치 시퀀스를 만든다 (논문 Alg 1의 CONSTRUCTBATCH를 에폭 단위로).

    emb_img/emb_txt: [N, D] L2 정규화 torch 텐서 (같은 device — GPU면 GPU에서 돈다).
    design/text: [N] 정수 라벨 torch 텐서.
    tau: LSE 온도 = 손실 온도 (1/logit_scale.exp()).
    반환: list[list[int]] — 각 배치의 레코드 인덱스. 자투리는 버린다.
    """
    device = emb_img.device
    n = emb_img.shape[0]
    b = int(batch_size)
    n_batches = n // b
    s = max(1, int(round(b * seed_frac)))
    k_pick = b - s
    tau = float(max(tau, 1e-4))            # 0 나누기 방지 (logit_scale clamp 100 → 0.01)

    g = generator if generator is not None else torch.Generator(device="cpu")
    remaining = torch.randperm(n, generator=g).to(device)   # U — 남은 인덱스

    batches = []
    for _ in range(n_batches):
        if remaining.numel() < b:
            break
        seeds, pool = remaining[:s], remaining[s:]

        # 후보 풀: 시드당 cross-modal hardness 상위 topk의 합집합 (논문 Alg 2 line 6)
        ti, vi = emb_txt[seeds], emb_img[seeds]              # [s, D]
        cross = (ti @ emb_img[pool].T + vi @ emb_txt[pool].T) / 2   # [s, |U|]
        k = min(topk, pool.numel())
        cand_pos = torch.unique(cross.topk(k, dim=1).indices.flatten())  # pool 내 위치
        cand = pool[cand_pos]                                # [C] 레코드 인덱스

        # w = cross − λ·intra − 이산 모순 페널티  (시드 × 후보)
        intra = (vi @ emb_img[cand].T + ti @ emb_txt[cand].T) / 2
        w = cross[:, cand_pos] - lam * intra
        same_text = text[seeds][:, None] == text[cand][None, :]
        same_design = design[seeds][:, None] == design[cand][None, :]
        contra = (same_text & ~same_design) if mask_false_negatives \
            else (same_text | same_design)
        w = w - penalty * contra.float()

        # LSE greedy (Eq 16). log-공간: logdenom_i = logsumexp_j w_ij/τ, j는 지금까지의 배치.
        # 시드끼리의 상호작용으로 초기화한다 (Eq 9의 합은 j ∈ B = S ∪ X 전체를 돈다).
        w_ss = ((ti @ vi.T + vi @ ti.T) / 2
                - lam * (vi @ vi.T + ti @ ti.T) / 2) / tau   # [s, s]
        logdenom = torch.logsumexp(w_ss, dim=1)              # [s]

        wt = w / tau                                         # [s, C]
        taken = torch.zeros(cand.numel(), dtype=torch.bool, device=device)
        picked = []
        for _ in range(min(k_pick, cand.numel())):
            # gain(v) = Σ_i τ·log(1 + exp(w_iv/τ − logdenom_i)) — softplus라 오버플로 없음
            gain = torch.nn.functional.softplus(wt - logdenom[:, None]).sum(0)
            gain[taken] = float("-inf")
            j = int(torch.argmax(gain))
            taken[j] = True
            picked.append(j)
            logdenom = torch.logaddexp(logdenom, wt[:, j])

        chosen = cand[torch.tensor(picked, device=device)]
        # 후보가 모자라면(중복 제목 밀집 등) 풀 앞쪽에서 무작위로 채워 배치 크기를 지킨다
        if chosen.numel() < k_pick:
            used = torch.zeros(n, dtype=torch.bool, device=device)
            used[chosen] = True
            filler = pool[~used[pool]][:k_pick - chosen.numel()]
            chosen = torch.cat([chosen, filler])

        batch = torch.cat([seeds, chosen])
        batches.append(batch.cpu().tolist())

        keep = torch.ones(n, dtype=torch.bool, device=device)
        keep[batch] = False
        remaining = pool[keep[pool]]
    return batches


class Hobit2BatchSampler:
    """DataLoader(batch_sampler=...)용. set_embeddings로 (이미지, 텍스트, τ)를 받는다.

    임베딩이 없으면(첫 갱신 전) 랜덤 배치로 폴백 — hobit.py와 같은 계약.
    배치 구성은 __iter__ 시작 시 에폭 전체를 한 번에 만든다(논문 Alg 1 line 5).
    임베딩이 GPU 텐서면 구성도 GPU에서 돈다.
    """

    def __init__(self, records, batch_size, topk=200, lam=1.0, seed_frac=0.25,
                 penalty=10.0, mask_false_negatives=True, seed=42):
        self.batch_size = int(batch_size)
        self.topk = int(topk)
        self.lam = float(lam)
        self.seed_frac = float(seed_frac)
        self.penalty = float(penalty)
        self.mask_fn = bool(mask_false_negatives)
        self.seed = int(seed)
        self.epoch = 0
        self.n = len(records)
        self.design = torch.from_numpy(
            _int_labels([r.get("design_id", r["image"]) for r in records]))
        self.text = torch.from_numpy(_int_labels([r.get("text", "") for r in records]))
        self.emb_img = self.emb_txt = None
        self.tau = 0.07                     # 첫 갱신 전 기본값 (logit_scale 초기값 근방)

    def set_embeddings(self, emb_img, emb_txt, tau):
        """[N, D] L2 정규화 임베딩 두 개와 현재 손실 온도. numpy면 torch로 변환."""
        for name, e in (("emb_img", emb_img), ("emb_txt", emb_txt)):
            if e is not None and len(e) != self.n:
                raise ValueError(f"{name} 행 수({len(e)})가 레코드 수({self.n})와 다르다")
        to_t = lambda e: torch.from_numpy(e) if isinstance(e, np.ndarray) else e
        self.emb_img, self.emb_txt = to_t(emb_img), to_t(emb_txt)
        self.tau = float(tau)

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        if self.emb_img is None or self.emb_txt is None:     # 폴백: 랜덤 배치
            perm = torch.randperm(self.n, generator=g)
            for k in range(len(self)):
                yield perm[k * self.batch_size:(k + 1) * self.batch_size].tolist()
            return
        device = self.emb_img.device
        batches = build_batches(
            self.emb_img, self.emb_txt, self.design.to(device), self.text.to(device),
            self.batch_size, self.tau, seed_frac=self.seed_frac, topk=self.topk,
            lam=self.lam, penalty=self.penalty, mask_false_negatives=self.mask_fn,
            generator=g)
        yield from batches
