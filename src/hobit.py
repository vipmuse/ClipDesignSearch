"""HOBIT: 에폭마다 submodular greedy로 미니배치를 재구성하는 배치 샘플러.

ICML 2026 "HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training"의 원리를
차용한다(전문 미확보 — 스펙 §1.3). 각 쿼리가 hard하되 모순되지 않는 네거티브를 보도록
배치를 구성하고, 목적함수가 monotone submodular라 greedy가 표준 근사 보장을 갖는다.

기존 PKBatchSampler가 휴리스틱(같은 로카르노 클래스로 묶기)으로 하던 일을 현재 모델의
임베딩에 근거해 수행한다. 모델은 알지 못하며 임베딩을 주입받는다 — GPU 없이 테스트하기
위해서이자, 임베딩 갱신 주기를 학습 루프가 결정하게 하기 위해서다.
"""
import numpy as np


def _int_labels(values):
    """동일 값 → 동일 정수. 배치 안 비교를 정수 비교로 싸게 만든다."""
    uniq = {}
    return np.array([uniq.setdefault(v, len(uniq)) for v in values], dtype=np.int64)


class HobitBatchSampler:
    """DataLoader(batch_sampler=...)용. 에폭마다 set_embeddings로 갱신된 임베딩을 쓴다.

    모순(contradiction)의 정의는 손실이 무엇을 positive로 보느냐에 달려 있다:
    - mask_false_negatives=True — 같은 design_id는 _pos_mask가 positive로 처리하므로
      모순이 아니다. 다른 design_id인데 텍스트가 같은 쌍만 모순이다.
    - mask_false_negatives=False — 대각선만 positive라 같은 design_id의 다른 뷰도
      네거티브로 밀린다. 둘 다 모순이다.

    한계: 에폭의 마지막 배치는 남은 인덱스가 정확히 batch_size개라 선택의 여지가 없다.
    모순이 있어도 그대로 들어간다. 모든 예제를 정확히 한 번 쓰는 한 구조적으로
    피할 수 없으며, 실데이터(에폭당 수천 배치)에서는 무시할 수준이다.
    """

    def __init__(self, records, batch_size, pool=4096, penalty=10.0,
                 mask_false_negatives=False, seed=42):
        self.batch_size = int(batch_size)
        self.pool = max(self.batch_size, int(pool))
        self.penalty = float(penalty)
        self.mask_fn = bool(mask_false_negatives)
        self.seed = int(seed)
        self.epoch = 0
        self.n = len(records)
        self.design = _int_labels([r.get("design_id", r["image"]) for r in records])
        self.text = _int_labels([r.get("text", "") for r in records])
        self.emb = None

    def set_embeddings(self, emb):
        """현재 모델의 [N, D] L2 정규화 임베딩. None이면 랜덤 배치로 폴백."""
        if emb is not None and len(emb) != self.n:
            raise ValueError(f"임베딩 행 수({len(emb)})가 레코드 수({self.n})와 다르다")
        self.emb = emb

    def __len__(self):
        """마지막 자투리(n % batch_size)는 버린다 — PKBatchSampler의 drop_last=True와 동일.

        greedy는 배치가 꽉 찬다는 전제로 gain을 계산하므로 부분 배치를 만들지 않는다.
        """
        return self.n // self.batch_size

    def _contra(self, cand, pick):
        """cand(후보 인덱스 배열) 각각이 pick(방금 배치에 넣은 인덱스)과 모순인지."""
        same_text = self.text[cand] == self.text[pick]
        same_design = self.design[cand] == self.design[pick]
        return (same_text & ~same_design) if self.mask_fn else (same_text | same_design)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1                              # 에폭마다 다른 배치
        n_batches = len(self)
        if n_batches == 0:
            return

        rem = rng.permutation(self.n)                # 남은 인덱스, 앞 m개가 유효
        m = self.n
        if self.emb is None:                         # 임베딩 없음 → 기존 랜덤 배치와 동일
            for k in range(n_batches):
                yield rem[k * self.batch_size:(k + 1) * self.batch_size].tolist()
            return

        for _ in range(n_batches):
            npool = min(self.pool, m)
            pos = rng.choice(m, size=npool, replace=False)   # rem 안에서의 위치
            cand = rem[pos]
            E = self.emb[cand]                       # [P, D]
            dots = np.zeros(npool, dtype="float32")  # 이미 담긴 것들과의 유사도 합
            contra = np.zeros(npool, dtype="float32")
            taken = np.zeros(npool, dtype=bool)

            chosen_pos = []
            first = 0                                # pos가 무작위라 0번이 곧 무작위 시드
            for slot in range(self.batch_size):
                if slot == 0:
                    j = first
                else:
                    gain = dots - self.penalty * contra
                    gain[taken] = -np.inf
                    j = int(np.argmax(gain))
                taken[j] = True
                chosen_pos.append(pos[j])
                dots += E @ E[j]                     # 증분 갱신 — 재계산 없이 O(P·D)
                contra += self._contra(cand, cand[j]).astype("float32")

            yield [int(rem[p]) for p in chosen_pos]

            for p in sorted(chosen_pos, reverse=True):   # 뒤에서부터 swap-remove
                m -= 1
                rem[p] = rem[m]
