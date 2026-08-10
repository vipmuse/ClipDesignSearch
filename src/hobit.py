"""HOBIT: 에폭마다 submodular greedy로 미니배치를 재구성하는 배치 샘플러.

ICML 2026 "HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training"의 원리를
차용한다(전문 미확보 — 스펙 §1.3). 각 쿼리가 hard하되 모순되지 않는 네거티브를 보도록
배치를 구성하고, 목적함수가 monotone submodular라 greedy가 표준 근사 보장을 갖는다.

기존 PKBatchSampler가 휴리스틱(같은 로카르노 클래스로 묶기)으로 하던 일을 현재 모델의
임베딩에 근거해 수행한다. 모델은 알지 못하며 임베딩을 주입받는다 — GPU 없이 테스트하기
위해서이자, 임베딩 갱신 주기를 학습 루프가 결정하게 하기 위해서다.
"""
import os

import numpy as np
from PIL import Image

from dataset import preprocess_drawing


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
    기본값 True는 train.py의 손실(mask_fn = t.get("mask_false_negatives", True))과 맞춘
    것이다. 키가 없을 때 손실과 샘플러가 서로 다른 것을 positive로 보면, 샘플러가
    손실의 정답 쌍을 피해 다니는(또는 그 반대) 조용한 불일치가 생긴다.

    한계 1 — 모순은 '마지막 배치' 하나가 아니라 꼬리로 쌓인다. penalty는 모순 쌍을
    없애는 게 아니라 뒤로 미루기만 하고, 모든 예제를 정확히 한 번 쓰는 커버리지 제약
    때문에 미뤄둔 것들은 에폭 끝에서 한꺼번에 소진된다. 실데이터 실측(425,140 레코드,
    penalty 10, pool 4096, batch 32 → 13,285 배치): 1~13,266번 배치는 모순 쌍이 정확히
    0이고 모순은 전부 마지막 19개 배치에 몰린다(2개는 200쌍 이상, 최대 496 = C(32,2)
    = 모든 쌍이 모순인 배치). 꼬리 길이는 제목 쏠림에 비례한다(최다 제목 점유율 3%면
    47배치, 16%면 795배치 — 실제 "Shoe"는 1.01%). 이 꼬리가 에폭의 끝, 즉 evaluate와
    어댑터 저장 바로 직전에 놓인다는 점에 유의한다.

    한계 2 — 메모리. self.emb는 학습 내내 상주한다(425,140 × 1024 float32 = 1.74 GB).
    refresh_embeddings는 교체본을 전부 만든 뒤 swap하므로 갱신 순간 호스트 RAM 피크는
    약 3.5 GB다.

    한계 3 — 재현성. 배치가 GPU에서 계산한 임베딩에 의존하므로 같은 seed의 두 hobit
    런이 같은 배치 시퀀스를 낸다는 보장이 없다. 시드 셔플/시드 PK로 완전히 결정적인
    다른 arm들과 달리 hobit의 Δ에는 런 간 분산이 더 섞여 있다 — 비교 시 감안할 것.
    """

    def __init__(self, records, batch_size, pool=4096, penalty=10.0,
                 mask_false_negatives=True, seed=42):
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


MAX_FAIL_RATIO = 0.01     # 이 비율을 넘게 실패하면 설정 사고로 보고 중단 (아래 근거)


class _DecodeCollator:
    """디코딩 + preprocess_drawing까지 워커에서 끝내는 collate.

    Windows spawn 워커로 피클되어야 하므로 람다·지역 클로저가 아닌 모듈 최상위 클래스다.
    반환 (PIL 리스트, 성공한 배치 내 위치, 배치 크기) — 실패를 건너뛰어도 호출자가
    원래 행 번호를 복원할 수 있어야 하고, 실패 수를 세려면 배치 크기도 필요하다.
    """

    def __init__(self, image_root, size):
        self.image_root = image_root
        self.size = size

    def __call__(self, batch):
        imgs, offs = [], []
        for k, r in enumerate(batch):
            try:
                im = Image.open(os.path.join(self.image_root, r["image"]))
                im.load()
                imgs.append(preprocess_drawing(im.convert("RGB"), self.size))
                offs.append(k)
            except Exception:
                continue                          # 0 벡터로 남는다 (정렬 유지)
        return imgs, offs, len(batch)


def _decode_serial(records, image_root, size, batch_size):
    """단일 프로세스 디코딩. (PIL 리스트, 행 번호, 배치 크기) 스트림."""
    coll = _DecodeCollator(image_root, size)
    for start in range(0, len(records), batch_size):
        imgs, offs, nb = coll(records[start:start + batch_size])
        yield imgs, [start + o for o in offs], nb


def _decode_parallel(records, image_root, size, batch_size, num_workers):
    """DataLoader 워커로 디코딩을 병렬화한 같은 스트림.

    디코딩+PIL 전처리는 실측 72 img/s(단일 스레드)인데 ViT-H forward는 배치 64 bf16에서
    254 img/s다 — 즉 갱신 비용의 78%가 GIL에 묶인 PIL 작업이다. 학습 루프가 이미 같은
    디코딩을 num_workers로 병렬화하고 있으므로, 손으로 스레드를 짜는 대신 같은
    PairDataset + DataLoader를 그대로 재사용한다.

    shuffle=False라 배치 순서가 sampler 순서와 같음이 보장된다 → 앞 배치들의 크기 합이
    곧 현재 배치의 시작 행 번호다.
    """
    from torch.utils.data import DataLoader

    from dataset import PairDataset
    loader = DataLoader(PairDataset(records), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        collate_fn=_DecodeCollator(image_root, size))
    start = 0
    for imgs, offs, nb in loader:
        yield imgs, [start + o for o in offs], nb
        start += nb


def embed_records(records, image_root, size, encode_fn, batch_size=64, num_workers=0,
                  max_fail_ratio=MAX_FAIL_RATIO):
    """레코드 순서를 보존한 [N, D] 임베딩. encode_fn(PIL 리스트) -> [B, D].

    행 i가 records[i]에 대응하는 것이 이 함수의 유일한 계약이다. 샘플러가 행 번호로
    design_id·텍스트를 조회하므로, 열 수 없는 이미지를 건너뛰면 그 뒤 전부가 밀려
    엉뚱한 레코드의 임베딩으로 배치를 짜게 된다. 실패 자리는 0 벡터로 채운다.

    num_workers>0이면 디코딩을 DataLoader 워커로 돌린다(기본 0 = 기존 직렬 경로).
    실데이터 425,140장 기준 직렬은 약 98.6분 — 그 뒤 에폭 자체보다 오래 걸린다.
    진행률은 tqdm으로 찍는다: 두 시간 동안 아무 출력이 없으면 운영자는 멈춘 줄 알고
    다일(多日) 학습을 죽인다.

    실패는 세어서 로그로 남기고, 조용히 성능만 나빠지는 대신 크게 실패한다:
      - 한 장도 못 열면 RuntimeError. (0 벡터 행렬을 돌려주면 set_embeddings가 행 수만
        보고 통과시키고, dots가 항상 0이라 greedy가 baseline 랜덤 배치로 퇴화한다.)
      - 실패 비율이 max_fail_ratio(기본 1%)를 넘어도 RuntimeError. 1%로 잡은 이유:
        472,615장 인덱스 빌드에서 실제로 못 연 도면은 극소수(0.1% 미만)라 정상 범위는
        1%에 한참 못 미치고, 반대로 --image-root 오지정·pairs.jsonl 경로 변경 같은
        설정 사고는 사실상 100% 실패로 나타난다. 그 사이에는 실데이터가 없어, 산발적
        손상 파일은 통과시키고 설정 사고는 잡는 경계로 1%를 고른다.
    """
    Image.MAX_IMAGE_PIXELS = None
    n = len(records)
    if n == 0:
        return np.zeros((0, 1), dtype="float32")

    from tqdm import tqdm
    stream = (_decode_parallel(records, image_root, size, batch_size, num_workers)
              if num_workers and num_workers > 0
              else _decode_serial(records, image_root, size, batch_size))
    out, failed = None, 0
    bar = tqdm(total=n, desc="hobit embed")
    try:
        for imgs, rows, nb in stream:
            failed += nb - len(imgs)
            if imgs:
                vec = np.asarray(encode_fn(imgs), dtype="float32")
                if out is None:
                    out = np.zeros((n, vec.shape[1]), dtype="float32")
                out[rows] = vec
            bar.update(nb)
    finally:
        bar.close()

    if failed:
        print(f"[hobit] 이미지 {failed}/{n}장을 열지 못해 0 벡터로 남겼다", flush=True)
    if out is None:
        raise RuntimeError(
            f"레코드 {n}개 중 임베딩에 성공한 이미지가 하나도 없다 — "
            f"image_root({image_root})와 pairs.jsonl의 상대경로를 확인할 것")
    if failed / n > max_fail_ratio:
        raise RuntimeError(
            f"임베딩 실패 {failed}/{n}장({failed / n:.1%})이 허용치 {max_fail_ratio:.1%}를 "
            f"넘는다 — image_root({image_root}) 오지정일 가능성이 높다. "
            f"실패 행은 0 벡터라 greedy가 그 레코드들을 에폭 끝으로 밀어낸다")
    return out


def refresh_embeddings(model, records, image_root, size, encode_fn, batch_size=64,
                       num_workers=0):
    """model을 eval로 내려 임베딩을 만들고, 예외가 나도 train 모드로 되돌린다.

    복구를 빠뜨리면 이후 학습이 조용히 eval 모드로 돌아 dropout이 꺼진 채 진행된다 —
    에러 없이 결과만 나빠지므로 반드시 finally로 보장한다.
    """
    model.eval()
    try:
        return embed_records(records, image_root, size, encode_fn, batch_size,
                             num_workers=num_workers)
    finally:
        model.train()
