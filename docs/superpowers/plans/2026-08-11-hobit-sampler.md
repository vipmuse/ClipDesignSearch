# HOBIT 배치 샘플러 구현 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 에폭마다 현재 모델 임베딩으로 submodular greedy 재정렬을 수행해, 각 쿼리가 "hard하되 모순되지 않는" 네거티브를 보도록 미니배치를 구성하는 `hobit` 학습 방법을 추가한다.

**Architecture:** 새 모듈 `src/hobit.py`가 두 가지를 제공한다 — 순수 numpy로 도는 `HobitBatchSampler`(임베딩을 주입받아 greedy로 배치를 채움)와 레코드 순서를 보존하는 임베딩 헬퍼 `embed_records`. `train.py`는 에폭 시작 시 현재 모델로 임베딩을 계산해 샘플러에 주입한다. 샘플러는 모델을 알지 못하므로 GPU 없이 단위 테스트할 수 있다.

**Tech Stack:** Python 3, numpy, PyTorch(cu128), transformers/peft, PyYAML, pytest

**근거 문서:** `docs/superpowers/specs/2026-08-10-training-method-comparison-design.md` §3.1

## Global Constraints

- 플랫폼은 Windows. 파이썬은 반드시 `.venv/Scripts/python.exe`를 쓴다. 전역 파이썬의 torch는 CPU 빌드라 사용 금지.
- 주석·로그·커밋 메시지는 한국어. 기존 코드 스타일(간결한 한국어 주석, 왜를 설명)을 따른다.
- 새 의존성을 추가하지 않는다.
- 테스트는 GPU·모델 로딩 없이 도는 것만 작성한다. 모델이 필요한 검증은 스모크로 미룬다.
- 메서드 정의는 `configs/methods/<name>.yaml` 한 장 + `src/registry.py` 병합이 유일한 경로다. 러너나 학습 코드에 방법별 분기를 하드코딩하지 않는다.
- 방법은 baseline 위에 **한 축만** 바꾼다. `hobit`이 바꾸는 축은 배치 구성뿐이다.

---

## 설계 결정 — 무엇이 "모순"인가

스펙 §3.1은 "같은 `design_id`이거나 동일 텍스트인 쌍"을 감점 대상으로 적었다. 그 문장은
`_pos_mask`가 텍스트 동일성을 positive로 취급하던 시점에 쓰였다. 지금은 다르다.

- `mask_false_negatives: true` — 같은 `design_id`는 `_pos_mask`가 이미 positive로 처리한다.
  손실이 이들을 밀어내지 않으므로 **모순이 아니다.** 감점하면 마스킹이 하려던 일을 되돌린다.
  이때 모순은 **다른 `design_id`인데 텍스트가 같은** 쌍뿐이다 (제목 중복 실측: 고유 제목
  28,859개 중 유일한 것 141개).
- `mask_false_negatives: false` — 대각선만 positive다. 같은 `design_id`의 다른 뷰도, 동일
  텍스트도 전부 네거티브로 밀린다. 둘 다 모순이다.

따라서 모순 판정은 `mask_false_negatives` 값에 따라 달라진다. 샘플러가 이 값을 인자로 받는다.

**hobit arm은 baseline 위에 sampler만 바꾸므로 `mask_false_negatives: false`다.** 즉 실제
비교에서 쓰이는 규칙은 후자다. 전자의 경로는 `all` 계열 레시피 위에 hobit을 얹을 때를 위해
지금 함께 구현한다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/hobit.py` (신규) | `HobitBatchSampler`, `embed_records`. 모델을 모른다 — 임베딩은 주입받는다 |
| `src/train.py` (수정) | `sampler` 설정값으로 샘플러 선택, 에폭 시작 시 임베딩 주입 |
| `configs/methods/hobit.yaml` (신규) | baseline + `sampler: hobit` |
| `tests/test_hobit.py` (신규) | 샘플러 성질과 임베딩 정렬 계약 |

`src/dataset.py`에 넣지 않는 이유: 이미 로더·분할·전처리·Collator·PK 샘플러를 담고 있어
책임이 넓다. HOBIT은 임베딩 의존성이라는 다른 성격의 결합을 갖는다.

---

### Task 1: `HobitBatchSampler` — greedy 배치 구성

**Files:**
- Create: `src/hobit.py`
- Create: `tests/test_hobit.py`

**Interfaces:**
- Consumes: 없음 (numpy만)
- Produces:
  - `HobitBatchSampler(records, batch_size, pool=4096, penalty=10.0, mask_false_negatives=False, seed=42)`
  - `.set_embeddings(emb: np.ndarray | None) -> None` — `[N, D]` L2 정규화 임베딩. `None`이면 랜덤 배치로 폴백
  - `__iter__` → 배치마다 `list[int]` (records 인덱스), `__len__` → 배치 수

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hobit.py` 생성:

```python
"""HOBIT 배치 샘플러: 커버리지·결정성·hardness·모순 회피를 고정."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hobit import HobitBatchSampler  # noqa: E402


def _recs(n_design=40, views=4, titles=("Shoe", "Bottle")):
    """디자인 n_design개 × 뷰 views장. 제목은 titles를 순환 → 다른 디자인이 같은 제목을 공유."""
    out = []
    for d in range(n_design):
        for v in range(views):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}",
                        "text": titles[d % len(titles)]})
    return out


def _two_clusters(n, dim=8, seed=0):
    """앞 절반과 뒤 절반이 서로 먼 두 군집. hardness가 작동하면 배치가 한 군집으로 뭉친다."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, dim)) * 0.05
    e[: n // 2, 0] += 1.0
    e[n // 2:, 0] -= 1.0
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype("float32")


def test_에폭당_인덱스가_중복없이_나온다():
    recs = _recs()                                  # 160 레코드
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    flat = [i for b in s for i in b]
    assert len(flat) == len(set(flat)), "같은 인덱스가 두 번 나왔다"
    assert len(flat) == (len(recs) // 16) * 16, "자투리 규약과 다르다"
    assert set(flat) <= set(range(len(recs)))


def test_모든_배치가_정확히_batch_size():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert all(len(b) == 16 for b in s)


def test_같은_seed_같은_에폭이면_같은_배치():
    recs = _recs()
    e = _two_clusters(len(recs))
    a = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); a.set_embeddings(e)
    b = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); b.set_embeddings(e)
    assert list(a) == list(b)


def test_에폭이_바뀌면_배치도_바뀐다():
    recs = _recs()
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7)
    s.set_embeddings(e)
    assert list(s) != list(s)                       # __iter__가 epoch을 증가시킨다


def test_임베딩이_없으면_랜덤_폴백하되_커버리지는_유지():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=3)
    flat = [i for b in s for i in b]                # set_embeddings 호출 없음
    assert len(flat) == len(set(flat)) == (len(recs) // 16) * 16


def test_hardness_배치가_한_군집으로_뭉친다():
    """무작위 배치라면 배치 내 평균 유사도가 0 근처. greedy가 작동하면 크게 양수."""
    recs = _recs(n_design=64, views=4)              # 256 레코드
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=0.0, seed=5)
    s.set_embeddings(e)
    sims = []
    for b in s:
        v = e[b]
        g = v @ v.T
        sims.append((g.sum() - np.trace(g)) / (len(b) * (len(b) - 1)))
    assert np.mean(sims) > 0.5, f"배치 내 유사도가 낮다 — greedy가 작동하지 않음: {np.mean(sims)}"


def test_마스킹_off면_같은_design_id를_같은_배치에_피한다():
    """mask_false_negatives=False면 같은 디자인의 뷰도 네거티브로 밀리므로 모순이다."""
    recs = _recs(n_design=64, views=4)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    dup = 0
    for b in s:
        ds = [recs[i]["design_id"] for i in b]
        dup += len(ds) - len(set(ds))
    assert dup == 0, f"같은 design_id가 한 배치에 {dup}번 겹쳤다"


def test_마스킹_on이면_같은_design_id는_허용하고_동일제목_타디자인만_피한다():
    recs = _recs(n_design=64, views=4)              # 제목 2종이 32개 디자인씩 공유
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    cross = 0
    for b in s:
        for x in range(len(b)):
            for y in range(x + 1, len(b)):
                rx, ry = recs[b[x]], recs[b[y]]
                if rx["text"] == ry["text"] and rx["design_id"] != ry["design_id"]:
                    cross += 1
    # 제목이 2종뿐이라 완전 회피는 불가능하지만, 무작위(배치16에서 약 60쌍)보다 훨씬 적어야 한다
    n_batches = len(recs) // 16
    assert cross / n_batches < 40, f"동일제목·타디자인 쌍이 배치당 {cross / n_batches:.1f}쌍"


def test_len은_배치_수와_일치():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert len(s) == len(list(s))
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hobit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hobit'`

- [ ] **Step 3: `src/hobit.py`의 샘플러 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hobit.py -v`
Expected: PASS (9 passed)

`test_hardness_배치가_한_군집으로_뭉친다`가 실패하면 greedy가 유사도를 최대화하지 않는 것이다. `dots` 증분 갱신의 부호와 `argmax` 방향을 확인한다.

- [ ] **Step 5: 전체 스위트 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 기존 73개 + 9개 = 82 passed

- [ ] **Step 6: 커밋**

```bash
git add src/hobit.py tests/test_hobit.py
git commit -m "HOBIT 배치 샘플러 추가 (submodular greedy)

현재 모델 임베딩으로 각 쿼리가 hard하되 모순되지 않는 네거티브를 보도록
배치를 greedy로 구성한다. 모순 판정은 mask_false_negatives에 따라 달라진다 —
마스킹이 켜져 있으면 같은 design_id는 이미 positive라 모순이 아니고,
다른 디자인인데 제목이 같은 쌍만 모순이다."
```

---

### Task 2: 임베딩 공급과 학습 루프 연결

**Files:**
- Modify: `src/hobit.py` (`embed_records` 추가)
- Modify: `src/train.py` (샘플러 선택 + 에폭 시작 시 임베딩 주입)
- Modify: `tests/test_hobit.py` (임베딩 정렬 계약 테스트 추가)

**Interfaces:**
- Consumes: `HobitBatchSampler.set_embeddings` (Task 1), `dataset.preprocess_drawing`
- Produces: `embed_records(records, image_root, size, encode_fn, batch_size=64) -> np.ndarray` — `[N, D]`. **행 i는 반드시 records[i]에 대응한다**; 열 수 없는 이미지는 0 벡터로 채워 정렬을 유지한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_hobit.py` 끝에 추가:

```python
from hobit import embed_records  # noqa: E402


def test_embed_records는_레코드_순서를_보존한다(tmp_path):
    from PIL import Image
    for i in range(5):
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(tmp_path / f"{i}.png")
    recs = [{"image": f"{i}.png"} for i in range(5)]

    def fake_encode(imgs):                       # 픽셀값을 그대로 벡터로
        return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")

    got = embed_records(recs, str(tmp_path), 16, fake_encode, batch_size=2)
    assert got.shape == (5, 3)
    assert [int(v[0]) for v in got] == [0, 10, 20, 30, 40]


def test_embed_records는_깨진_이미지를_0벡터로_채워_정렬을_유지한다(tmp_path):
    from PIL import Image
    Image.new("RGB", (16, 16), (10, 0, 0)).save(tmp_path / "ok.png")
    recs = [{"image": "missing.png"}, {"image": "ok.png"}]

    def fake_encode(imgs):
        return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")

    got = embed_records(recs, str(tmp_path), 16, fake_encode, batch_size=2)
    assert got.shape == (2, 3)
    assert np.all(got[0] == 0), "열 수 없는 이미지 자리가 0 벡터가 아니다"
    assert int(got[1][0]) == 10, "행이 밀려 레코드와 어긋났다"
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hobit.py -v`
Expected: FAIL — `ImportError: cannot import name 'embed_records'`

- [ ] **Step 3: `embed_records` 구현**

`src/hobit.py` 상단 import에 추가:

```python
import os

from PIL import Image

from dataset import preprocess_drawing
```

파일 끝에 추가:

```python
def embed_records(records, image_root, size, encode_fn, batch_size=64):
    """레코드 순서를 보존한 [N, D] 임베딩. encode_fn(PIL 리스트) -> [B, D].

    행 i가 records[i]에 대응하는 것이 이 함수의 유일한 계약이다. 샘플러가 행 번호로
    design_id·텍스트를 조회하므로, 열 수 없는 이미지를 건너뛰면 그 뒤 전부가 밀려
    엉뚱한 레코드의 임베딩으로 배치를 짜게 된다. 실패 자리는 0 벡터로 채운다.
    """
    Image.MAX_IMAGE_PIXELS = None
    out, buf, buf_rows, dim = None, [], [], None

    def flush():
        nonlocal out, dim
        if not buf:
            return
        vec = np.asarray(encode_fn(buf), dtype="float32")
        if out is None:
            dim = vec.shape[1]
            out = np.zeros((len(records), dim), dtype="float32")
        out[buf_rows] = vec
        buf.clear(); buf_rows.clear()

    for i, r in enumerate(records):
        try:
            im = Image.open(os.path.join(image_root, r["image"]))
            im.load()
            buf.append(preprocess_drawing(im.convert("RGB"), size))
            buf_rows.append(i)
        except Exception:
            continue                              # 0 벡터로 남는다 (정렬 유지)
        if len(buf) >= batch_size:
            flush()
    flush()
    return out if out is not None else np.zeros((len(records), 1), dtype="float32")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hobit.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: `train.py`에 샘플러 선택과 임베딩 주입 연결**

`src/train.py`의 import에 추가:

```python
from hobit import HobitBatchSampler, embed_records
```

샘플러 선택부(`pk_views = t.get("pk_views", 4)`로 시작하는 블록)를 아래로 교체:

```python
    sampler = None
    which = t.get("sampler", "pk" if t.get("pk_views", 4) > 1 else "random")
    if which == "hobit":
        sampler = HobitBatchSampler(
            train_recs, t["batch_size"], pool=t.get("hobit_pool", 4096),
            penalty=t.get("hobit_penalty", 10.0),
            mask_false_negatives=t.get("mask_false_negatives", True), seed=t["seed"])
    elif which == "pk":
        sampler = PKBatchSampler(train_recs, t["batch_size"],
                                 views_per_design=t.get("pk_views", 4),
                                 locarno_aware=t.get("locarno_aware", True), seed=t["seed"])
    if sampler is not None:
        train_loader = DataLoader(PairDataset(train_recs), batch_sampler=sampler,
                                  num_workers=t["num_workers"], collate_fn=collate_train)
    else:                                  # 기존 랜덤 배치 (baseline)
        train_loader = DataLoader(PairDataset(train_recs), batch_size=t["batch_size"],
                                  shuffle=True, num_workers=t["num_workers"],
                                  collate_fn=collate_train, drop_last=True)
```

에폭 루프(`for epoch in range(t["epochs"]):`) 바로 다음 줄, `if stop: break` 앞에 임베딩 갱신을 넣는다:

```python
    for epoch in range(t["epochs"]):
        if isinstance(sampler, HobitBatchSampler) and \
                epoch % max(1, t.get("hobit_refresh_every", 1)) == 0:
            # 배치 구성이 "현재" 모델 기준이어야 hard negative가 의미를 갖는다.
            # 학습 집합 전체를 1회 추론하는 비용이 에폭마다 든다 → refresh_every로 조절.
            model.eval()
            with torch.no_grad():
                emb = embed_records(
                    train_recs, args.image_root, cfg["model"]["image_size"],
                    lambda imgs: _encode_for_hobit(model, processor, imgs, device, dtype))
            model.train()
            sampler.set_embeddings(emb)
            print(f"[hobit] epoch {epoch}: 임베딩 {emb.shape} 갱신", flush=True)
        if stop:
            break
```

`_pos_mask` 위에 인코딩 헬퍼를 추가한다:

```python
@torch.no_grad()
def _encode_for_hobit(model, processor, imgs, device, dtype):
    """HOBIT 배치 구성용 이미지 임베딩 [B, D] (L2 정규화). 증강 없이 결정적으로."""
    px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        emb = model.get_image_features(pixel_values=px)
    if not torch.is_tensor(emb):          # MetaCLIP 2는 출력 객체를 반환
        emb = emb.pooler_output
    emb = F.normalize(emb.float(), dim=-1)
    return emb.cpu().numpy().astype("float32")
```

- [ ] **Step 6: 전체 스위트 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 73 + 11 = 84 passed

- [ ] **Step 7: 커밋**

```bash
git add src/hobit.py src/train.py tests/test_hobit.py
git commit -m "HOBIT 임베딩 공급과 학습 루프 연결

에폭 시작 시 현재 모델로 학습 집합을 인코딩해 샘플러에 주입한다. embed_records는
행 i가 records[i]에 대응하는 것을 계약으로 삼는다 — 열 수 없는 이미지를 건너뛰면
그 뒤가 전부 밀려 엉뚱한 레코드로 배치를 짜게 되므로 0 벡터로 채운다.
train.sampler 설정값으로 random/pk/hobit을 고른다."
```

---

### Task 3: `hobit` 메서드 등록과 문서 갱신

**Files:**
- Create: `configs/methods/hobit.yaml`
- Modify: `tests/test_method_configs.py` (기본 arm 구별 테스트에 영향 없는지 확인 + hobit 전용 단언)
- Modify: `ACCURACY.md` (§3-1 arm 표에 hobit 추가)

**Interfaces:**
- Consumes: `registry.resolve` / `registry.list_methods` (기존), `HobitBatchSampler` (Task 1), `train.py`의 `sampler` 선택자 (Task 2)
- Produces: 메서드 이름 `hobit`. `run_ablation.py --arms baseline hobit`으로 실행 가능.

- [ ] **Step 1: 메서드 YAML 작성**

`configs/methods/hobit.yaml`:

```yaml
name: hobit
description: "에폭마다 submodular greedy 재정렬로 배치 구성 (baseline + 배치 구성 축만 변경)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  sampler: hobit
  hobit_pool: 4096
  hobit_penalty: 10.0
  hobit_refresh_every: 1
  pk_views: 1
  locarno_aware: false
  mask_false_negatives: false
  augment: false
  img2img_weight: 0.0
```

`pk_views` 이하 다섯 값은 `baseline.yaml`과 동일하다. hobit이 바꾸는 것은 `sampler`뿐이므로,
비교표에서 Δ가 곧 배치 구성의 기여가 된다.

- [ ] **Step 2: 메서드 등록 확인 테스트 추가**

`tests/test_method_configs.py` 끝에 추가:

```python
def test_hobit은_baseline과_sampler만_다르다():
    """hobit이 바꾸는 축은 배치 구성 하나여야 Δ가 그 기여로 읽힌다."""
    h = registry.resolve("hobit")["train"]
    b = registry.resolve("baseline")["train"]
    five = ("pk_views", "locarno_aware", "mask_false_negatives", "augment", "img2img_weight")
    assert [h[k] for k in five] == [b[k] for k in five]
    assert h["sampler"] == "hobit"
    assert b.get("sampler", "pk") != "hobit"


def test_hobit_하이퍼파라미터가_존재한다():
    h = registry.resolve("hobit")["train"]
    assert h["hobit_pool"] >= h["batch_size"], "후보 풀이 배치보다 작으면 greedy가 무의미"
    assert h["hobit_penalty"] > 0
    assert h["hobit_refresh_every"] >= 1
```

- [ ] **Step 3: 테스트 실행**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_method_configs.py -v`
Expected: PASS. 파라미터화 테스트가 메서드 11개를 덮는다.

`test_기본_arm들은_서로_다른_레시피다`는 `DEFAULT_ARMS`만 보므로 hobit 추가에 영향받지 않는다. 실패하면 그 테스트가 `registry.list_methods()`를 쓰고 있는지 확인하고, `DEFAULT_ARMS` 기준으로 되돌린다.

- [ ] **Step 4: `ACCURACY.md` §3-1 arm 표에 hobit 추가**

"선택 arm" 표 바로 위, 기본 arm 표 끝에 행을 추가한다:

```
| `hobit` | baseline + submodular greedy 배치 구성 | ① 배치 구성의 원리적 대체 (ICML 2026) |
```

같은 절의 실행 예시 아래에 한 줄 덧붙인다:

```
python scripts\run_ablation.py --arms baseline hobit --epochs 3   # 배치 구성 단독 기여
```

- [ ] **Step 5: 스모크 — 실제로 학습이 도는지**

Run:

```bash
./.venv/Scripts/python.exe scripts/run_ablation.py --arms hobit --limit 400 --epochs 1 --max-steps 5 --eval-batches 1 --no-index --force
```

Expected: `[hobit] epoch 0: 임베딩 (N, 1024) 갱신`이 찍히고 5스텝 학습 후 평가까지 진행된다.

실패 시 확인 순서: (a) `sampler: hobit`이 resolved config에 실려 있는지, (b) `embed_records`가 `[N, 1024]`를 반환하는지, (c) `set_embeddings`의 행 수 검증에 걸리는지. CUDA OOM이면 `--limit`을 200으로 줄인다. 환경 문제로 완주하지 못하면 무엇을 대신 검증했는지 기록한다.

- [ ] **Step 6: 전체 스위트 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 86 passed

- [ ] **Step 7: 커밋**

```bash
git add configs/methods/hobit.yaml tests/test_method_configs.py ACCURACY.md
git commit -m "hobit 메서드 등록 — baseline 대비 배치 구성 축만 변경

sampler 외 다섯 개 토글을 baseline과 동일하게 두어, 비교표의 Δ가 곧 배치 구성의
기여가 되도록 한다. 이를 테스트로 고정한다."
```

---

## 완료 기준

- `./.venv/Scripts/python.exe -m pytest tests/ -v` 전부 통과 (약 86개)
- `run_ablation.py --arms baseline hobit`이 두 arm을 같은 홀드아웃에서 학습·평가
- `hobit`이 baseline과 `sampler` 하나만 다르다는 것이 테스트로 고정됨

## 스펙과의 의도적 편차 2건

- **스펙 §5는 "에폭당 모든 예제가 정확히 한 번"을 테스트 대상으로 적었으나, 마지막
  자투리(`n % batch_size`)는 버린다.** greedy가 배치가 꽉 찬다는 전제로 gain을 계산하고,
  기존 `PKBatchSampler`도 `drop_last=True`가 기본이라 조건을 맞춘 것이다. 테스트는 "중복
  없음 + 정확히 `(n // batch_size) * batch_size`개"로 고정한다.
- **스펙 §2 예시 YAML의 `hobit_score: cosine` 선택자를 두지 않는다.** 지금 선택지가 하나뿐인
  설정 키는 쓰이지 않는 분기만 늘린다. 논문 입수 시 교체 지점은 `HobitBatchSampler`의
  `dots += E @ E[j]` 한 줄이며, 그때 선택자가 필요하면 그 시점에 추가한다.

## 알려진 비용과 한계

- **에폭마다 학습 집합 전체를 1회 추론한다.** 10만 장 부분집합에서 ViT-H 기준 에폭당 10분
  내외가 추가된다. `hobit_refresh_every`를 2~3으로 올리면 비례해 줄지만, 배치가 "현재"
  모델 기준이라는 전제는 그만큼 약해진다.
- **논문 재현이 아니라 원리 차용이다** (스펙 §1.3). hardness 스코어는 배치 내 코사인
  유사도 합으로 정의했고, 논문의 정의는 확보하지 못했다. 교체 지점은 `HobitBatchSampler`의
  `dots` 갱신 한 줄이다.
- `hobit_penalty`의 기본 10.0은 임베딩이 L2 정규화라 유사도 항이 배치 크기(32) 규모를
  넘지 않는다는 점에 근거한 값이며, 실측으로 조정한 값은 아니다.

## 다음 단계

Phase 3의 나머지 네 방법(`tic`, `bigbatch`, `hires384`, `loracap`)은 각각 별도 계획으로
다룬다. `loracap`은 `lora` 블록 오버라이드만으로 끝나므로 가장 짧고, `bigbatch`(GradCache)와
`hires384`(모델 교체)가 가장 무겁다.
