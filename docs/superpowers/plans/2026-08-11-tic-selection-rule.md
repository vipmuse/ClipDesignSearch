# TIC 선택 규칙 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TIC이 밀어낼 쌍을 고르는 규칙을 코사인 임계값 하나에서 **헤드명사 + 코사인 상한** 두 축으로 바꿔, 실제로 `Pizza box`/`Storage box` 같은 목표 쌍에 발동하고 `Clothing hanger`/`Clothes hanger` 같은 표기 차이는 건드리지 않게 한다.

**Architecture:** `Collator`가 제목의 헤드명사(마지막 알파벳 단어) 정수 라벨을 배치에 추가하고, `tic_loss`의 eligible 마스크가 그 라벨과 코사인 상한을 함께 본다. 손실 형태(hinge)·게이트·로깅·`compose_loss` 구조는 그대로 재사용한다.

**Tech Stack:** Python 3, PyTorch(cu128), transformers/peft, PyYAML, pytest

**근거 문서:** `docs/superpowers/specs/2026-08-10-training-method-comparison-design.md` §3.2

## Global Constraints

- 플랫폼은 Windows. 파이썬은 반드시 `.venv/Scripts/python.exe`를 쓴다. 전역 파이썬의 torch는 CPU 빌드라 사용 금지.
- 주석·로그·커밋 메시지는 한국어. 기존 코드 스타일(간결한 한국어 주석, 왜를 설명)을 따른다.
- 새 의존성을 추가하지 않는다.
- 테스트는 GPU·모델 로딩 없이 도는 것만 작성한다. 모델이 필요한 검증은 스모크로 미룬다.
- `text_label`은 절대 positive 판정에 쓰지 않는다. `head_label`도 마찬가지다 — 둘 다 필터 전용이다.
- `tic`은 baseline 위에 손실 축 하나만 바꾼다.

---

## 왜 바꾸는가 — 실측 기록

1차 설계는 "코사인이 `margin`(0.9)보다 높은 쌍을 밀어낸다"였고 실측으로 반증됐다.

```
목표 쌍인데 margin 아래라 발동 안 함:  Container/Beverage container 0.786, Shoe/Ballet shoe 0.801
무관한데 그보다 높음:                  Shoe/Bottle 0.867
margin을 넘는 것들(붙어 있어야 맞음):  Eyeglasses/Glasses 0.951, Wash basin/Washbasin 0.968
결과: TIC 항이 CLIP 손실 3.5에 4e-6 기여 (배치당 대상 496쌍 중 0.35쌍만 발동)
```

원인은 스칼라 하나에 **물품군 선택**과 **표기차 배제** 두 가지 일을 시킨 것이다. 두 축으로 나눈다.

**같은 헤드명사 쌍 40만 표본 실측** (베이스 모델):

```
분포: p50 0.765 · p75 0.824 · p90 0.870 · p95 0.896 · p99 0.950 · ≥0.92는 2.45%
상한 위(제외 대상): 'Portable Bluetooth speaker'/'Portable bluetooth speaker' 0.997
                   'Clothing hanger'/'Clothes hanger' 0.996
                   'Handheld vacuum cleaner'/'Hand-held vacuum cleaner' 0.994
0.78~0.88(목표군): 'Wine carrier'/'Pet carrier', 'Pizza box'/'Storage box',
                   'Button panel'/'Wall panel', 'Flow sensor'/'Sensor'
배치(32) 대상 쌍: 평균 3.03, 중앙값 2, 최대 16, 0개인 배치 10.3%
                   (기존 495.7쌍 대비 165배 감소 → .mean() 분모가 줄어 쌍당 가중치가 커진다)
```

`floor`는 분포 중앙값 0.75, `ceiling`은 0.92로 둔다. `tic_margin`은 없어지고 두 값이 대신한다.

---

## 파일 구조

| 파일 | 변경 |
|---|---|
| `src/dataset.py` | `Collator`가 `head_label` 추가 |
| `src/train.py` | `tic_loss` eligible 규칙 교체, `compose_loss`가 `head_label` 전달, `main()`의 파라미터 캐스팅 갱신 |
| `configs/methods/tic.yaml` | `tic_margin` → `tic_floor`/`tic_ceiling` |
| `tests/test_text_label.py` | `head_label` 테스트 추가 |
| `tests/test_tic.py` | 규칙 테스트 교체·추가 |
| `tests/test_method_configs.py` | 범위 검증을 새 키로 |

---

### Task 1: `head_label` 추가

**Files:**
- Modify: `src/dataset.py` (`Collator.__call__`)
- Modify: `tests/test_text_label.py`

**Interfaces:**
- Consumes: 없음
- Produces: 배치에 `head_label` (`torch.long [B]`). 같은 헤드명사 → 같은 정수. 원문 제목 기준.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_text_label.py` 끝에 추가:

```python
def test_같은_헤드명사면_같은_라벨(tmp_path):
    """헤드명사 = 제목의 마지막 알파벳 단어. TIC이 '같은 물품군'을 좁히는 축이다."""
    for i in range(4):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    recs = _records(["Pizza box", "Storage box", "Wine carrier", "Shoe"],
                    ["A", "B", "C", "D"])
    hl = _collate(recs, str(tmp_path))["head_label"].tolist()
    assert hl[0] == hl[1], "box끼리 같은 라벨이어야 한다"
    assert hl[0] != hl[2] and hl[0] != hl[3]


def test_헤드명사는_대소문자와_구두점을_무시한다(tmp_path):
    for i in range(3):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    recs = _records(["Storage Box", "pizza box.", "Wall panel"], ["A", "B", "C"])
    hl = _collate(recs, str(tmp_path))["head_label"].tolist()
    assert hl[0] == hl[1]
    assert hl[0] != hl[2]


def test_헤드명사도_원문_기준이다(tmp_path):
    """증강이 viewpoint를 뒤에 붙이면 마지막 단어가 바뀐다 — 원문으로 계산해야 한다."""
    for i in range(2):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    recs = _records(["Pizza box", "Storage box"], ["A", "B"],
                    viewpoints=["front view", "side view"])
    for _ in range(30):
        hl = _collate(recs, str(tmp_path), augment=True)["head_label"].tolist()
        assert hl[0] == hl[1], "증강된 텍스트로 헤드명사를 뽑고 있다"


def test_알파벳이_없는_제목은_빈_헤드로_묶인다(tmp_path):
    for i in range(2):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    hl = _collate(_records(["123", "456"], ["A", "B"]), str(tmp_path))["head_label"].tolist()
    assert hl[0] == hl[1]
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_text_label.py -v`
Expected: FAIL — `KeyError: 'head_label'` (앞 세 개), 네 번째도 같은 이유로 실패

- [ ] **Step 3: `Collator`가 `head_label`을 만들도록 수정**

`src/dataset.py`에는 `re`가 import되어 있지 않다 — 상단 표준 라이브러리 import 블록(`json`/`math`/`os`/`random`)에 알파벳 순서에 맞게 `import re`를 추가한다. 그다음 모듈 수준에 헬퍼를 넣는다:

```python
def head_noun(text):
    """제목의 헤드명사 = 마지막 알파벳 단어(소문자). 없으면 빈 문자열.

    'Pizza box'와 'Storage box'를 같은 물품군으로 묶는 축. TIC이 이 축으로 후보를
    좁힌 뒤 코사인 상한으로 표기 차이('Clothing hanger'/'Clothes hanger')를 뺀다.
    """
    w = re.findall(r"[a-z]+", text.lower())
    return w[-1] if w else ""
```

`Collator.__call__`에서 `text_label`을 만드는 블록 아래에 추가:

```python
        # 같은 헤드명사 → 같은 정수. text_label과 같이 TIC 필터 전용이며,
        # 증강 전 원문(raw_texts)으로 계산한다 — 증강은 뒤에 viewpoint를 붙여
        # 마지막 단어를 바꿔버린다.
        uniq_h = {}
        enc["head_label"] = torch.tensor(
            [uniq_h.setdefault(head_noun(t), len(uniq_h)) for t in raw_texts],
            dtype=torch.long)
```

`Collator` 독스트링에 한 줄 더한다:

```python
    head_label: 같은 헤드명사(제목 마지막 단어) → 같은 정수. TIC 필터 전용.
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_text_label.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: 전체 스위트 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 141 passed, 11 skipped

- [ ] **Step 6: 커밋**

```bash
git add src/dataset.py tests/test_text_label.py
git commit -m "Collator에 head_label 추가 — TIC의 물품군 축

제목의 마지막 알파벳 단어를 정수 라벨로. 'Pizza box'와 'Storage box'를
같은 물품군으로 묶어 TIC이 밀어낼 후보를 좁힌다. text_label과 마찬가지로
필터 전용이고 증강 전 원문으로 계산한다."
```

---

### Task 2: `tic_loss`의 선택 규칙 교체

**Files:**
- Modify: `src/train.py` (`tic_loss`, `compose_loss`, `main()`의 파라미터 캐스팅, 학습 루프 호출부)
- Modify: `tests/test_tic.py`

**Interfaces:**
- Consumes: `head_label` (Task 1)
- Produces:
  - `tic_loss(text_embeds, design_label, text_label, head_label, floor=0.75, ceiling=0.92, return_stats=False)`
  - `compose_loss(out, pos, t, design_label, text_label, head_label) -> (loss, stats)`

- [ ] **Step 1: 기존 테스트를 새 시그니처로 고치고 규칙 테스트를 추가**

`tests/test_tic.py`에서 `tic_loss(...)`를 호출하는 기존 테스트들은 인자가 늘어난다. 각 호출에 `head_label`을 추가하고 `margin=` 대신 `floor=`/`ceiling=`을 쓴다. 예를 들어 `test_제목이_같으면_밀어내지_않는다`는 이렇게 된다:

```python
def test_제목이_같으면_밀어내지_않는다():
    """같은 문자열 → 같은 임베딩. 분리가 불가능하므로 손실 0이어야 한다."""
    e = _emb([[1.0, 0.0], [1.0, 0.0]])
    assert tic_loss(e, torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 0]),
                    floor=0.5).item() == 0.0
```

나머지 기존 테스트도 같은 방식으로 `head_label`을 **같은 값**(같은 물품군)으로 주고 `margin=0.5`를 `floor=0.5`로 바꾼다. 그리고 새 규칙 테스트를 추가한다:

```python
def test_헤드명사가_다르면_밀어내지_않는다():
    """'Shoe'와 'Bottle'은 코사인이 높아도 서로 다른 물품군이라 대상이 아니다.
    1차 설계가 이 쌍을 밀어내려 한 것이 실패의 한 축이었다."""
    e = _emb([[1.0, 0.0], [1.0, 0.05]])              # 유사도 ≈ 1.0
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
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tic.py -v`
Expected: FAIL — `TypeError: tic_loss() got an unexpected keyword argument 'floor'` 등

- [ ] **Step 3: `tic_loss` 교체**

`src/train.py`의 `tic_loss` 시그니처와 마스크를 바꾼다. 독스트링의 설명도 새 규칙에 맞춘다:

```python
def tic_loss(text_embeds, design_label, text_label, head_label,
             floor=0.75, ceiling=0.92, return_stats=False):
    """텍스트 모달 내부 대조(TIC). 같은 물품군의 '서로 다른 물품'만 밀어낸다.

    선택 규칙은 두 축이다 (2026-08-11 실측으로 확정, 스펙 §3.2):
    - 같은 헤드명사: 'Pizza box'/'Storage box'처럼 관련 물품군으로 후보를 좁힌다.
      이 축이 없으면 'Shoe'/'Bottle'(코사인 0.867)까지 대상이 된다.
    - 코사인 < ceiling: 'Clothing hanger'/'Clothes hanger'(0.996)처럼 같은 물품의
      표기 차이를 뺀다. 이들은 붙어 있어야 맞다.
    여기에 문자열이 같은 쌍(같은 벡터라 분리 불가)과 같은 design_id 쌍을 더 뺀다.

    스칼라 임계값 하나로는 안 되는 이유: 베이스 모델 코사인은 물품 유사도 순으로
    정렬되지 않는다. 목표 쌍 Container/Beverage container는 0.786인데 무관한
    Shoe/Bottle이 0.867이고, 0.9를 넘는 것은 Eyeglasses/Glasses 같은 동의어다.
    물품군 선택과 표기차 배제는 서로 다른 축이라 하나로 겸할 수 없다.

    floor 위로 올라온 만큼만 hinge로 민다. floor 0.75는 같은 헤드명사 쌍 분포의
    중앙값이라, 배치당 대상 약 3쌍 중 절반쯤이 실제로 밀린다.

    return_stats=True면 (loss, 대상 쌍 수, 위반 쌍 수)를 돌려준다. 대상 쌍 수는
    상한을 적용한 뒤 기준이다 — 필터가 얼마나 걸러냈는지 로그에서 보이도록.
    """
    # .float(): bf16 autocast 아래서는 t @ t.t()도 bf16이라 임계 근처 표현 간격이
    # ~0.002다. F.normalize는 오늘의 MetaCLIP 2 출력에는 중복이지만 지우면 안 된다:
    # 이 손실을 행 단위 스케일 불변으로 만들어, 밀어내기가 노름을 부풀리거나
    # 무너뜨리는 방향으로 새지 않게 한다.
    t = F.normalize(text_embeds.float(), dim=-1)
    sim = t @ t.t()
    B = t.size(0)
    upper = torch.triu(torch.ones(B, B, dtype=torch.bool, device=t.device), diagonal=1)
    eligible = upper & (design_label[:, None] != design_label[None, :]) \
                     & (text_label[:, None] != text_label[None, :]) \
                     & (head_label[:, None] == head_label[None, :]) \
                     & (sim < ceiling)
    if not eligible.any():
        zero = t.new_tensor(0.0)
        return (zero, 0, 0) if return_stats else zero
    hinge = (sim[eligible] - floor).clamp(min=0)
    loss = hinge.mean()
    return (loss, int(eligible.sum()), int((hinge > 0).sum())) if return_stats else loss
```

- [ ] **Step 4: `compose_loss`와 호출부 갱신**

`compose_loss`의 시그니처에 `head_label`을 더하고 TIC 호출을 바꾼다:

```python
def compose_loss(out, pos, t, design_label, text_label, head_label):
```

```python
    if t.get("tic_weight", 0.0) > 0:
        tic, n_el, n_vi = tic_loss(out.text_embeds, design_label, text_label, head_label,
                                   floor=t.get("tic_floor", 0.75),
                                   ceiling=t.get("tic_ceiling", 0.92), return_stats=True)
```

학습 루프에서 `head_label`을 꺼내 넘긴다 (`text_label`을 꺼내는 줄 옆):

```python
            head_label = enc.pop("head_label").to(device)
```

그리고 `compose_loss(out, pos, t, design_label, text_label, head_label)`로 호출한다.

`evaluate()`에서도 버린다 (`text_label`을 버리는 줄 옆):

```python
        enc.pop("head_label", None)                    # 평가에는 쓰지 않는다
```

`main()`의 파라미터 캐스팅에서 `tic_margin`을 새 키 두 개로 바꾼다:

```python
    if "tic_weight" in t:                      # 문자열로 적혀도 학습 중간이 아니라 지금 터지게
        t["tic_weight"] = float(t["tic_weight"])
        t["tic_floor"] = float(t.get("tic_floor", 0.75))
        t["tic_ceiling"] = float(t.get("tic_ceiling", 0.92))
```

시작 로그도 새 값으로 바꾼다:

```python
        print(f"[tic] ON — tic_weight={t['tic_weight']} "
              f"floor={t['tic_floor']} ceiling={t['tic_ceiling']}", flush=True)
```

`tests/test_tic.py`의 `compose_loss` 관련 테스트들도 `head_label` 인자를 받도록 고친다. 게이트 테스트에서 쓰는 라벨은 **같은 헤드명사**로 줘야 TIC이 실제로 발동한다:

```python
    head_label = torch.tensor([0, 0])
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tic.py -v`
Expected: PASS. 실패하면 `head_label`을 넘기지 않은 호출부가 남은 것이다.

- [ ] **Step 6: 전체 스위트 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 전부 통과. `tic_margin`을 참조하는 `tests/test_method_configs.py`가 실패하면 Task 3에서 고치므로, 그 실패만 남는 것은 정상이다 — 어떤 테스트가 왜 실패하는지 리포트에 적는다.

- [ ] **Step 7: 커밋**

```bash
git add src/train.py tests/test_tic.py
git commit -m "TIC 선택 규칙을 헤드명사+코사인 상한 두 축으로 교체

스칼라 margin 하나로는 목표 쌍을 고를 수 없다는 것이 실측으로 확인됐다.
Container/Beverage container 0.786은 margin 아래인데 무관한 Shoe/Bottle이
0.867이고, margin을 넘는 것은 Eyeglasses/Glasses 같은 동의어였다.
헤드명사로 물품군을 좁히고 상한으로 표기 차이를 뺀다."
```

---

### Task 3: `tic.yaml` 파라미터 교체와 가중치 실측 결정

**Files:**
- Modify: `configs/methods/tic.yaml`
- Modify: `tests/test_method_configs.py`
- Modify: `ACCURACY.md`

**Interfaces:**
- Consumes: `tic_loss`의 `floor`/`ceiling` (Task 2)
- Produces: 메서드 `tic`의 최종 파라미터

- [ ] **Step 1: YAML 파라미터 교체**

`configs/methods/tic.yaml`의 `train` 블록에서 `tic_margin`을 지우고 두 키를 넣는다. `tic_weight`는 Step 4에서 실측으로 정하므로 일단 1.0으로 둔다:

```yaml
train:
  tic_weight: 1.0
  tic_floor: 0.75
  tic_ceiling: 0.92
  pk_views: 1
  locarno_aware: false
  mask_false_negatives: false
  augment: false
  img2img_weight: 0.0
```

- [ ] **Step 2: 범위 검증 테스트를 새 키로 갱신**

`tests/test_method_configs.py`에서 `tic_margin`을 참조하는 단언을 바꾼다:

```python
    assert "tic_floor" in t, "tic_floor 누락 — 키 오타면 기본값으로 조용히 넘어간다"
    assert "tic_ceiling" in t, "tic_ceiling 누락"
    assert 0 < t["tic_floor"] < t["tic_ceiling"] < 1, \
        f"floor < ceiling < 1 이어야 한다: floor={t['tic_floor']} ceiling={t['tic_ceiling']}"
    assert t["tic_weight"] > 0
```

`tic_*` 접두사 화이트리스트가 있는 테스트도 새 키 이름으로 갱신한다.

- [ ] **Step 3: 테스트 실행**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 전부 통과

- [ ] **Step 4: 가중치를 실측으로 정한다 — 이 태스크의 핵심**

`tic_weight`를 추측으로 정하지 않는다. 1차 설계가 실패한 이유 중 하나가 0.2라는 근거 없는 값이었고, 리뷰 실측상 전체 쌍이 위반해도 0.2의 그래디언트 비는 0.004에 그쳤다.

`scripts/run_ablation.py`는 `--log-every`를 자식에게 전달하지 않는다(전달하는 것은 `--epochs`/`--limit`/`--max-steps`/`--eval-batches`뿐). `config`의 `log_every`가 20이라 20스텝 실행에서는 로그가 한 줄만 나온다. 그러므로 러너를 거치지 않고 `src/train.py`를 직접 돌린다 — `train.py`에는 `--log-every`가 있다.

먼저 resolved config를 만든다:

```bash
./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import registry; print(registry.write_resolved('tic'))"
```

그다음 **반드시 Bash 툴의 `run_in_background: true`로** 학습을 띄운다 (포그라운드로 띄우면 턴 종료 시 프로세스 트리가 정리되어 로그가 0바이트로 남는다 — 앞선 방법 구현에서 두 번 겪은 실패다):

```bash
PYTHONUNBUFFERED=1 ./.venv/Scripts/python.exe src/train.py \
  --config outputs/methods/tic/config.resolved.yaml \
  --limit 3000 --epochs 1 --max-steps 20 --eval-batches 1 --log-every 1
```

로그에서 다음을 읽는다:

```
[tic] ON — tic_weight=1.0 floor=0.75 ceiling=0.92
e0 s1/20 loss=... tic=<가중 전 값> (×w→<가중 후>) 위반/대상=<n>/<m>
```

**판단 기준**: 가중 후 TIC 항이 총 손실의 **1~3%**가 되도록 `tic_weight`를 정한다. 예를 들어 총 손실 3.5에 가중 후 TIC이 0.002라면 목표(0.035~0.105)까지 약 20~50배가 필요하므로 `tic_weight`를 20~50으로 올린다. 계산 과정을 리포트에 그대로 적는다.

또한 **대상 쌍 수가 실측 예측(평균 3, 0개인 배치 10%)과 맞는지** 확인한다. 크게 다르면 `head_label`이나 상한이 의도대로 동작하지 않는다는 뜻이므로 그 사실을 리포트에 적고 멈춘다.

`--limit 3000`을 쓰는 이유: 400개면 배치당 대상 쌍이 너무 적어 20스텝으로는 표본이 부족하다.

- [ ] **Step 5: 정해진 가중치를 YAML에 반영하고 재확인**

Step 4에서 정한 값으로 `tic_weight`를 고친 뒤, 같은 train.py 명령을 다시 돌려 가중 후 TIC 항이 목표 구간에 들어오는지 확인한다. 로그 두 줄(변경 전/후)을 리포트에 남긴다.

- [ ] **Step 6: `ACCURACY.md` 갱신**

`tic` 행의 설명을 새 규칙으로 바꾼다:

```
| (선택) `tic` | baseline + 텍스트 모달 내부 대조(헤드명사+상한 선택) | ③ 같은 물품군의 다른 물품이 텍스트 공간에서 구분되지 않음 |
```

- [ ] **Step 7: 전체 스위트 확인 후 커밋**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add configs/methods/tic.yaml tests/test_method_configs.py ACCURACY.md
git commit -m "tic 파라미터를 floor/ceiling으로 교체하고 가중치를 실측으로 결정

tic_margin 하나를 tic_floor(0.75, 같은 헤드명사 쌍 분포 중앙값)와
tic_ceiling(0.92, 표기 차이 배제)으로 나눈다. tic_weight는 추측하지 않고
20스텝 로그에서 TIC 항이 총 손실의 1~3%가 되도록 실측해 정했다."
```

---

## 완료 기준

- `./.venv/Scripts/python.exe -m pytest tests/ -v` 전부 통과
- 로그의 가중 후 TIC 항이 총 손실의 1~3% 구간에 있고, 대상 쌍 수가 배치당 평균 3 내외
- `tic`이 baseline과 손실 축 하나만 다르다는 기존 테스트가 계속 통과

## 알려진 한계

- **신호가 희소하다.** 배치당 대상 쌍이 평균 3개, 10%의 배치는 0개다. 정규화 항으로는
  받아들일 만하지만 그래디언트가 매 스텝 들어오지는 않는다.
- **상한 0.92는 베이스 모델 기준이다.** 학습이 진행되면 텍스트 공간이 움직여 상한이 다른
  것을 걸러낼 수 있다. 로그의 대상/위반 쌍 수가 학습 중 어떻게 변하는지가 관측 지점이다.
- **헤드명사는 마지막 단어라는 근사다.** 'Portion of a shoe'는 헤드가 'shoe'로 맞게 잡히지만
  'Shoe with laces'는 'laces'가 된다. 전체 분포에서 이 오류가 얼마나 되는지는 재지 않았다.
