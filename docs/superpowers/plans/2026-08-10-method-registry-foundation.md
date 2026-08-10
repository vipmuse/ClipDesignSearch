# 메서드 레지스트리 기반 구축 (Phase 0~2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5개 학습 방법을 같은 조건에서 비교할 수 있도록, 망가진 baseline을 고치고 메서드 레지스트리와 방법별 인덱스 파이프라인을 만든다.

**Architecture:** `configs/methods/<name>.yaml` 한 장이 방법 하나를 규정하고, `src/registry.py`가 유일한 병합 지점이 되어 `outputs/methods/<name>/config.resolved.yaml`을 못박는다. 이후 학습·평가·인덱스 단계는 이 resolved 파일만 읽으므로 병합 로직이 어긋날 수 없다. 인덱스에는 지문(`index_meta.json`)을 남겨 어댑터와 짝이 맞는지 검증 가능하게 한다.

**Tech Stack:** Python 3, PyTorch(cu128), transformers/peft, faiss-cpu, PyYAML, pytest(신규)

**근거 문서:** `docs/superpowers/specs/2026-08-10-training-method-comparison-design.md`

## Global Constraints

- 플랫폼은 Windows. 파이썬은 반드시 `.venv/Scripts/python.exe`를 쓴다. 전역 파이썬의 torch는 CPU 빌드라 사용 금지.
- 심볼릭 링크를 쓰지 않는다 (관리자 권한 필요). 파일 참조는 포인터 JSON으로 표현한다.
- 주석·로그·커밋 메시지는 한국어. 기존 코드 스타일(간결한 한국어 주석, 왜를 설명)을 따른다.
- 새 런타임 의존성을 추가하지 않는다. 개발 의존성으로 `pytest`만 추가한다.
- 테스트는 GPU·모델 로딩 없이 도는 것만 작성한다. 모델이 필요한 검증은 e2e 스모크로 미룬다.
- 기존 `outputs/index`, `outputs/lora-clip-design`는 건드리지 않는다. 신규 산출물은 전부 `outputs/methods/` 아래로 간다.

---

### Task 1: positive 마스크 정상화 + pytest 도입

**배경:** `_pos_mask`가 "텍스트가 같으면 positive"로 판정한다. 실측상 고유 제목 28,859개 중 유일한 것이 141개뿐이라 `Shoe`(4,720건) 같은 명칭이 대량 반복된다. `locarno_aware: true`와 겹치면 배치 32개가 모두 같은 제목이 되어 `pos_mask`가 사실상 전부 True → `masked_clip_loss`가 변별 그래디언트를 잃는다. 이 상태에서는 이후 모든 방법 비교가 무의미하다.

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_pos_mask.py`
- Modify: `src/train.py:64-67` (`_pos_mask`), `src/train.py:87-89` (`evaluate`), `src/train.py:189-190` (학습 루프)
- Modify: `src/dataset.py:190-193` (`Collator`의 `text_label` 제거)

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `_pos_mask(design_label: torch.Tensor) -> torch.Tensor` — 인자 1개로 바뀐다. 이후 태스크는 `text_label`이 배치에 없다고 가정해도 된다.

- [ ] **Step 1: pytest를 개발 의존성으로 추가**

`requirements.txt` 맨 아래에 추가:

```
pytest          # 개발 전용 (테스트 실행)
```

설치:

```bash
./.venv/Scripts/python.exe -m pip install pytest
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_pos_mask.py` 생성:

```python
"""positive 마스크 회귀 테스트.

제목 동일성을 positive 근거로 삼으면 안 된다는 것을 고정한다. 실측(2026-08)상
고유 제목 28,859개 중 유일한 것이 141개뿐이라, 제목 기반 positive는 서로 다른
디자인을 정답으로 묶어 학습 신호를 없앤다.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import _pos_mask  # noqa: E402


def test_동일_제목_32개가_전부_다른_디자인이면_대각선만_positive():
    # 'Shoe' 32건이 한 배치에 모인 최악의 경우를 모사
    design = torch.arange(32)
    pos = _pos_mask(design)
    assert torch.equal(pos, torch.eye(32, dtype=torch.bool))
    assert pos.sum().item() == 32


def test_같은_design_id의_뷰끼리는_positive():
    design = torch.tensor([0, 0, 1, 1, 2])
    pos = _pos_mask(design)
    assert pos[0, 1] and pos[1, 0]
    assert not pos[0, 2]
    assert pos.sum().item() == 9          # 2x2 블록 두 개 + 단독 1


def test_대칭이고_대각선을_포함한다():
    design = torch.tensor([3, 3, 7])
    pos = _pos_mask(design)
    assert torch.equal(pos, pos.t())
    assert pos.diagonal().all()
```

- [ ] **Step 3: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_pos_mask.py -v`
Expected: FAIL — `TypeError: _pos_mask() missing 1 required positional argument: 'text_label'`

- [ ] **Step 4: `_pos_mask`를 design_id 기준으로 수정**

`src/train.py:64-67`을 아래로 교체:

```python
def _pos_mask(design_label):
    """레코드 i,j가 같은 design_id면 positive. 대각선 포함, 대칭.

    제목 동일성은 positive 근거가 되지 못한다. 고유 제목 28,859개 중 유일한 것이
    141개뿐이라(2026-08 실측) 'Shoe'가 4,720건 반복된다. 제목이 같다고 묶으면 서로
    다른 디자인이 정답으로 취급돼, 특히 locarno_aware 배치에서 마스크가 거의 전부
    True가 되고 변별 그래디언트가 사라진다.
    """
    return design_label[:, None] == design_label[None, :]
```

- [ ] **Step 5: 호출부 2곳에서 `text_label` 제거**

`src/train.py:87-89` (`evaluate` 안):

```python
        d = enc.pop("design_label").to(device)
        pos = _pos_mask(d)                             # 대칭 → 양방향 공용
```

`src/train.py:189-190` (학습 루프 안) — `text_label`을 꺼내던 줄을 지우고:

```python
            design_label = enc.pop("design_label").to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            # 마스킹 비활성 시 대각선만 positive = 표준 CLIP InfoNCE와 동일
            pos = _pos_mask(design_label) if mask_fn \
                else torch.eye(design_label.size(0), dtype=torch.bool, device=device)
```

- [ ] **Step 6: `Collator`에서 `text_label` 생성 제거**

`src/dataset.py`에서 아래 3줄을 삭제한다 (`enc["design_label"]` 다음 블록):

```python
        # 동일 텍스트 → 같은 정수 라벨 (InfoNCE false-negative 마스킹용)
        uniq_t = {}
        enc["text_label"] = torch.tensor(
            [uniq_t.setdefault(t, len(uniq_t)) for t in texts], dtype=torch.long)
```

같은 파일 `Collator` 독스트링에서 `text_label` 언급을 지운다:

```python
    design_label: 같은 design_id를 positive로 묶는 멀티-positive loss
    (masked InfoNCE, supcon)용 정수 라벨.
```

이 삭제로 별도 결함 하나가 함께 사라진다 — `text_label`이 증강된 텍스트(30% 확률로 viewpoint가 붙은)로 계산돼 같은 제목이 배치마다 다른 라벨을 받던 문제다.

- [ ] **Step 7: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_pos_mask.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: 학습 루프가 실제로 도는지 스모크**

Run: `./.venv/Scripts/python.exe src/train.py --limit 512 --max-steps 5 --eval-batches 2`
Expected: 예외 없이 5스텝 진행. `KeyError: 'text_label'`이 나면 지우지 못한 호출부가 남은 것이다.

- [ ] **Step 9: 커밋**

```bash
git add requirements.txt tests/test_pos_mask.py src/train.py src/dataset.py
git commit -m "positive 마스크를 design_id 기준으로 정상화 + pytest 도입

제목 동일성 positive는 고유 제목 141/28,859 실측상 거의 항상 발동해
locarno_aware 배치와 겹치면 마스크가 전부 True가 되고 학습 신호가 사라졌다.
text_label 자체를 제거해 증강 텍스트로 라벨이 흔들리던 문제도 함께 해소."
```

---

### Task 2: `--limit` 표본 편향 제거 + 홀드아웃 재현성

**배경:** `records[:limit]`은 인제스천 순서 앞부분만 잘라내므로 특허번호·로카르노가 한 구간에 몰린 표본이 된다. `run_ablation.py --quick`이 정확히 `--limit 2000`을 쓰므로 스모크 비교가 무의미해진다. 또한 `eval_retrieval.py`가 `args.seed or t["seed"]`로 되어 있어 `--seed 0`을 명시해도 무시된다 — 홀드아웃 재현이 목적인 스크립트에서 치명적이다.

**Files:**
- Modify: `src/dataset.py` (신규 함수 `take_limit` 추가)
- Modify: `src/train.py:137-139`
- Modify: `src/eval_retrieval.py:66-67, 74-76`
- Create: `tests/test_take_limit.py`

**Interfaces:**
- Consumes: 없음
- Produces: `take_limit(records: list[dict], limit: int, seed: int) -> list[dict]` — `src/dataset.py`. 학습과 평가가 **반드시 같은 함수를 같은 seed로** 호출해야 동일 표본을 본다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_take_limit.py` 생성:

```python
"""--limit 축소 표본이 인제스천 순서에 편향되지 않는지 고정."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import take_limit  # noqa: E402


def _recs(n):
    # 앞 절반은 로카르노 0101, 뒤 절반은 0202 — 순서대로 자르면 한쪽만 뽑힌다
    return [{"image": f"{i}.png", "locarno": "0101" if i < n // 2 else "0202"}
            for i in range(n)]


def test_앞에서_자르지_않고_전_구간에서_뽑는다():
    got = take_limit(_recs(1000), 100, seed=42)
    codes = {r["locarno"] for r in got}
    assert codes == {"0101", "0202"}, "한쪽 구간만 뽑혔다 — 셔플이 빠졌다"
    assert len(got) == 100


def test_같은_seed는_같은_표본():
    a = take_limit(_recs(1000), 100, seed=42)
    b = take_limit(_recs(1000), 100, seed=42)
    assert [r["image"] for r in a] == [r["image"] for r in b]


def test_다른_seed는_다른_표본():
    a = take_limit(_recs(1000), 100, seed=1)
    b = take_limit(_recs(1000), 100, seed=2)
    assert [r["image"] for r in a] != [r["image"] for r in b]


def test_limit이_0이거나_전체보다_크면_원본_그대로():
    recs = _recs(50)
    assert take_limit(recs, 0, seed=42) is recs
    assert take_limit(recs, 100, seed=42) is recs
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_take_limit.py -v`
Expected: FAIL — `ImportError: cannot import name 'take_limit'`

- [ ] **Step 3: `take_limit` 구현**

`src/dataset.py`의 `load_records` 아래에 추가 (파일 상단에 `import random`이 이미 있는지 확인하고 없으면 추가):

```python
def take_limit(records, limit, seed):
    """--limit 축소 표본. 앞에서 자르면 인제스천 순서(특허번호·로카르노가 몰린 구간)만
    보게 되므로 seed 고정 셔플 후 자른다. 원래 순서는 유지해 재현 시 진단이 쉽도록 한다.

    train.py와 eval_retrieval.py가 같은 seed로 이 함수를 써야 동일 표본을 본다.
    """
    if not limit or limit >= len(records):
        return records
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    return [records[i] for i in sorted(idx[:limit])]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_take_limit.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: `train.py`가 `take_limit`을 쓰도록 수정**

`src/train.py`의 import에 `take_limit` 추가:

```python
from dataset import Collator, PairDataset, PKBatchSampler, load_records, split_by_design, take_limit
```

`src/train.py:137-139`를 교체:

```python
    records = load_records(t["data_path"])
    records = take_limit(records, args.limit, t["seed"])
```

- [ ] **Step 6: `eval_retrieval.py`의 seed 처리와 limit 수정**

`src/eval_retrieval.py:66-67`의 두 인자를 `default=None`으로 바꾼다:

```python
    ap.add_argument("--eval-ratio", type=float, default=None, help="기본: config eval_ratio")
    ap.add_argument("--seed", type=int, default=None, help="기본: config seed")
```

`src/eval_retrieval.py:74-76`을 교체 (`0`을 명시해도 무시되던 문제 해소):

```python
    ratio = args.eval_ratio if args.eval_ratio is not None else t["eval_ratio"]
    seed = args.seed if args.seed is not None else t["seed"]
```

import에 `take_limit`을 추가하고, `records = records[:args.limit]` 부분을 교체:

```python
    records = load_records(args.data or t["data_path"])
    records = take_limit(records, args.limit, seed)
```

- [ ] **Step 7: 전체 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS (7 passed)

- [ ] **Step 8: 커밋**

```bash
git add src/dataset.py src/train.py src/eval_retrieval.py tests/test_take_limit.py
git commit -m "--limit 표본 편향 제거 + 홀드아웃 seed 재현성 수정

앞에서 자르던 --limit이 인제스천 순서 한 구간만 보게 만들어 --quick ablation
비교가 무의미했다. take_limit()으로 학습·평가가 같은 표본을 보게 통일.
eval_retrieval의 --seed 0 / --eval-ratio 0이 무시되던 문제도 함께 수정."
```

---

### Task 3: 메서드 레지스트리 로더

**배경:** 학습·평가·인덱스·서빙이 각자 YAML을 병합하면 네 곳의 병합 로직이 어긋난다. 병합을 이 모듈 하나로 모으고 결과를 파일로 못박는다.

**Files:**
- Create: `src/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `deep_merge(base: dict, over: dict) -> dict`
  - `list_methods() -> list[str]`
  - `method_dir(name: str) -> str` — `outputs/methods/<name>` 절대경로
  - `resolve(name: str, base_config: str | None = None) -> dict` — 병합된 config dict
  - `write_resolved(name: str, cfg: dict | None = None) -> str` — `config.resolved.yaml` 경로 반환
  - `write_data_pointer(name: str, cfg: dict) -> str` — `outputs/methods/<name>/data/pairs.jsonl` 포인터 파일을 쓰고, 실제로 읽어야 할 pairs 경로를 반환

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_registry.py` 생성:

```python
"""메서드 레지스트리: 병합이 결정적이고 진실원천이 하나임을 고정."""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import registry  # noqa: E402


@pytest.fixture
def 임시_레지스트리(tmp_path, monkeypatch):
    """configs/methods 와 outputs/methods 를 tmp_path로 돌린 격리 환경."""
    methods = tmp_path / "configs" / "methods"
    methods.mkdir(parents=True)
    base = tmp_path / "configs" / "lora_clip.yaml"
    base.write_text(yaml.safe_dump({
        "model": {"model_id": "models/base", "image_size": 224},
        "lora": {"r": 16, "alpha": 32},
        "train": {"batch_size": 32, "seed": 42, "augment": False, "output_dir": "outputs/x"},
    }), encoding="utf-8")
    (methods / "demo.yaml").write_text(yaml.safe_dump({
        "name": "demo",
        "description": "테스트용",
        "extends": "configs/lora_clip.yaml",
        "data": {"builder": "shared", "pairs": "data/pairs.jsonl"},
        "train": {"augment": True, "sampler": "hobit"},
    }), encoding="utf-8")
    monkeypatch.setattr(registry, "ROOT", str(tmp_path))
    monkeypatch.setattr(registry, "METHODS_DIR", str(methods))
    monkeypatch.setattr(registry, "OUT_ROOT", str(tmp_path / "outputs" / "methods"))
    return tmp_path


def test_deep_merge는_중첩_딕셔너리를_병합한다():
    got = registry.deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert got == {"a": {"x": 1, "y": 9}, "b": 3}


def test_deep_merge는_원본을_바꾸지_않는다():
    base = {"a": {"x": 1}}
    registry.deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_resolve는_extends_베이스에_오버라이드를_얹는다(임시_레지스트리):
    cfg = registry.resolve("demo")
    assert cfg["model"]["model_id"] == "models/base"     # 베이스에서 옴
    assert cfg["train"]["batch_size"] == 32              # 베이스에서 옴
    assert cfg["train"]["augment"] is True               # 오버라이드
    assert cfg["train"]["sampler"] == "hobit"            # 오버라이드가 추가한 키


def test_resolve는_method_블록과_output_dir를_채운다(임시_레지스트리):
    cfg = registry.resolve("demo")
    assert cfg["method"]["name"] == "demo"
    assert cfg["method"]["data"]["builder"] == "shared"
    assert cfg["train"]["output_dir"] == "outputs/methods/demo"


def test_resolve는_결정적이다(임시_레지스트리):
    assert registry.resolve("demo") == registry.resolve("demo")


def test_모르는_메서드는_에러(임시_레지스트리):
    with pytest.raises(FileNotFoundError):
        registry.resolve("없는방법")


def test_list_methods는_정렬된_이름을_준다(임시_레지스트리, tmp_path):
    (tmp_path / "configs" / "methods" / "alpha.yaml").write_text(
        yaml.safe_dump({"name": "alpha"}), encoding="utf-8")
    assert registry.list_methods() == ["alpha", "demo"]


def test_write_resolved가_쓴_파일은_resolve_결과와_같다(임시_레지스트리):
    path = registry.write_resolved("demo")
    assert os.path.basename(path) == "config.resolved.yaml"
    written = yaml.safe_load(open(path, encoding="utf-8"))
    assert written == registry.resolve("demo")


def test_shared_데이터는_포인터_파일로_공용_pairs를_가리킨다(임시_레지스트리):
    cfg = registry.resolve("demo")
    actual = registry.write_data_pointer("demo", cfg)
    assert actual.replace("\\", "/").endswith("data/pairs.jsonl")
    pointer = os.path.join(registry.method_dir("demo"), "data", "pairs.jsonl.pointer.json")
    assert os.path.exists(pointer), "포인터 파일이 생성되지 않았다"
    import json
    assert json.load(open(pointer, encoding="utf-8"))["source"] == "data/pairs.jsonl"
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'registry'`

- [ ] **Step 3: `src/registry.py` 구현**

```python
"""메서드 레지스트리: configs/methods/<name>.yaml 한 장이 방법 하나를 규정한다.

병합은 이 모듈에서만 한다. 학습·평가·인덱스·서빙이 각자 YAML을 병합하면 네 곳의
병합 로직이 어긋나 조용히 다른 설정으로 도는 사고가 난다. resolve()가 만든
config.resolved.yaml만 이후 단계가 읽는다 — 재현성도 여기서 나온다.
"""
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS_DIR = os.path.join(ROOT, "configs", "methods")
OUT_ROOT = os.path.join(ROOT, "outputs", "methods")

DEFAULT_BASE = "configs/lora_clip.yaml"
_OVERRIDABLE = ("model", "lora", "train")     # 메서드 YAML이 덮어쓸 수 있는 최상위 블록


def deep_merge(base, over):
    """over를 base 위에 재귀 병합한 새 dict. base는 변경하지 않는다."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) \
            if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def list_methods():
    """등록된 메서드 이름 목록 (정렬)."""
    if not os.path.isdir(METHODS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(METHODS_DIR) if f.endswith(".yaml"))


def method_dir(name):
    return os.path.join(OUT_ROOT, name)


def resolve(name, base_config=None):
    """메서드 YAML을 extends 베이스와 병합한 dict. 파일 쓰기는 하지 않는다."""
    path = os.path.join(METHODS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"알 수 없는 메서드: {name} ({path})")
    spec = yaml.safe_load(open(path, encoding="utf-8")) or {}
    base_rel = base_config or spec.get("extends", DEFAULT_BASE)
    base = yaml.safe_load(open(os.path.join(ROOT, base_rel), encoding="utf-8"))

    cfg = deep_merge(base, {k: v for k, v in spec.items() if k in _OVERRIDABLE})
    cfg["method"] = {
        "name": name,
        "description": spec.get("description", ""),
        "extends": base_rel,
        "data": spec.get("data", {"builder": "shared", "pairs": "data/pairs.jsonl"}),
    }
    # 산출물 경로는 규약으로 고정 — 메서드 YAML이 지정하지 못하게 한다
    cfg["train"]["output_dir"] = os.path.relpath(method_dir(name), ROOT).replace("\\", "/")
    return cfg


def write_resolved(name, cfg=None):
    """resolved config를 outputs/methods/<name>/config.resolved.yaml로 못박고 경로 반환."""
    cfg = resolve(name) if cfg is None else cfg
    d = method_dir(name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "config.resolved.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def write_data_pointer(name, cfg):
    """방법의 데이터 위치를 산출물 디렉터리에 기록하고 실제 pairs 경로를 반환한다.

    builder가 shared면 472k 레코드를 복제하지 않고 포인터 JSON만 남긴다. 심볼릭
    링크를 쓰지 않는 이유는 Windows에서 관리자 권한이 필요하기 때문이다. 향후
    전용 데이터를 만드는 builder가 생기면 여기서 실제 파일 경로를 반환하면 된다.
    """
    data = cfg.get("method", {}).get("data", {})
    source = data.get("pairs", "data/pairs.jsonl")
    d = os.path.join(method_dir(name), "data")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pairs.jsonl.pointer.json"), "w", encoding="utf-8") as f:
        json.dump({"builder": data.get("builder", "shared"), "source": source},
                  f, ensure_ascii=False, indent=2)
    return os.path.join(ROOT, source)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "메서드 레지스트리 로더 추가

configs/methods/<name>.yaml + extends 병합을 이 모듈 한 곳에서만 수행하고
config.resolved.yaml로 못박는다. 이후 학습·평가·인덱스·서빙은 resolved만 읽어
병합 로직이 네 곳으로 갈라지는 것을 막는다."
```

---

### Task 4: 기존 arm을 메서드 YAML로 이관하고 러너를 레지스트리 기반으로 전환

**배경:** `run_ablation.py`의 `ARMS` 딕셔너리가 방법 정의의 두 번째 진실원천이 되어 있다. 이를 YAML로 옮기고, 러너가 `config.resolved.yaml`을 만들어 **학습과 평가 양쪽에 같은 config를 넘기도록** 한다. 지금은 평가에 `--config`가 전달되지 않아 `--config`를 지정해도 평가만 `configs/lora_clip.yaml`을 읽는 결함이 있다.

**Files:**
- Create: `configs/methods/baseline.yaml`, `aug.yaml`, `mask.yaml`, `pkmask.yaml`, `pkmask-i2i.yaml`, `all.yaml`
- Modify: `scripts/run_ablation.py` (전면 개편)
- Create: `tests/test_method_configs.py`

**Interfaces:**
- Consumes: `registry.resolve`, `registry.write_resolved`, `registry.write_data_pointer`, `registry.list_methods`, `registry.method_dir`, `registry.OUT_ROOT` (Task 3)
- Produces:
  - `train_arm(name, args) -> tuple[str, str]` — `(어댑터 경로, config.resolved.yaml 경로)`
  - `eval_arm(name, adapter, cfg_path, args) -> None`
  - 산출물 경로 규약 `outputs/methods/<name>/{config.resolved.yaml, data/, final/, eval/}`

- [ ] **Step 1: 메서드 YAML 6개 작성**

`configs/methods/baseline.yaml`:

```yaml
name: baseline
description: "모든 개선 OFF — 기존 파이프라인과 동일 조건. 비교의 기준점"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 1
  locarno_aware: false
  mask_false_negatives: false
  augment: false
  img2img_weight: 0.0
```

`configs/methods/aug.yaml`:

```yaml
name: aug
description: "+ 도면 증강 (소회전·스케일·라인두께·viewpoint 텍스트)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 1
  locarno_aware: false
  mask_false_negatives: false
  augment: true
  img2img_weight: 0.0
```

`configs/methods/mask.yaml`:

```yaml
name: mask
description: "+ false-negative 마스킹 InfoNCE (design_id 기준)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 1
  locarno_aware: false
  mask_false_negatives: true
  augment: false
  img2img_weight: 0.0
```

`configs/methods/pkmask.yaml`:

```yaml
name: pkmask
description: "+ PK 샘플러 + 마스킹 (세트 — PK 단독은 false negative를 늘려 유해)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 4
  locarno_aware: true
  mask_false_negatives: true
  augment: false
  img2img_weight: 0.0
```

`configs/methods/pkmask-i2i.yaml`:

```yaml
name: pkmask-i2i
description: "+ PK + 마스킹 + img2img supcon (supcon은 PK 없이는 미발화)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 4
  locarno_aware: true
  mask_false_negatives: true
  augment: false
  img2img_weight: 0.5
```

`configs/methods/all.yaml`:

```yaml
name: all
description: "전부 ON (= configs/lora_clip.yaml 기본값)"
extends: configs/lora_clip.yaml
data: {builder: shared, pairs: data/pairs.jsonl}
train:
  pk_views: 4
  locarno_aware: true
  mask_false_negatives: true
  augment: true
  img2img_weight: 0.5
```

- [ ] **Step 2: 메서드 YAML 무결성 테스트 작성**

`tests/test_method_configs.py` 생성:

```python
"""등록된 모든 메서드 YAML이 실제로 resolve되는지 확인.

YAML 오타나 extends 경로 오류를 학습 몇 시간 뒤가 아니라 지금 잡는다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import registry  # noqa: E402

METHODS = registry.list_methods()


def test_메서드가_하나_이상_등록되어_있다():
    assert METHODS, "configs/methods/*.yaml 이 비어 있다"


@pytest.mark.parametrize("name", METHODS)
def test_모든_메서드가_resolve된다(name):
    cfg = registry.resolve(name)
    assert cfg["method"]["name"] == name
    for key in ("model", "lora", "train"):
        assert key in cfg, f"{name}: {key} 블록 누락"
    assert cfg["train"]["output_dir"] == f"outputs/methods/{name}"


@pytest.mark.parametrize("name", METHODS)
def test_YAML의_name_필드가_파일명과_일치한다(name):
    import yaml
    spec = yaml.safe_load(open(os.path.join(registry.METHODS_DIR, f"{name}.yaml"),
                               encoding="utf-8"))
    assert spec.get("name") == name, f"{name}.yaml 의 name 필드가 파일명과 다르다"
```

- [ ] **Step 3: 테스트를 돌려 통과를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_method_configs.py -v`
Expected: PASS — 메서드 6개 × 2 + 1 = 13 passed. 실패하면 해당 YAML의 오타를 고친다.

- [ ] **Step 4: 러너를 레지스트리 기반으로 개편**

`scripts/run_ablation.py`의 `OFF`/`ALL_ON`/`_arm`/`ARMS`/`DEFAULT_ARMS`/`deep_merge` 정의를 삭제하고, 상단 import와 경로를 아래로 바꾼다:

```python
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import registry  # noqa: E402

ABL_DIR = registry.OUT_ROOT
DEFAULT_ARMS = ["baseline", "aug", "mask", "pkmask", "pkmask-i2i", "all"]
```

`eval_json_path`를 규약에 맞춘다:

```python
def eval_json_path(name):
    fname = "base.json" if name == "base" else "final.json"
    return os.path.join(registry.method_dir(name), "eval", fname)
```

`train_arm`을 교체 — config 생성을 레지스트리에 위임하고 어댑터 경로를 규약으로 고정:

```python
def train_arm(name, args):
    """resolved config와 데이터 포인터를 못박고 학습. (어댑터 경로, config 경로) 반환."""
    adir = registry.method_dir(name)
    adapter = os.path.join(adir, "final")
    cfg = registry.resolve(name)
    cfg_path = registry.write_resolved(name, cfg)
    registry.write_data_pointer(name, cfg)          # 산출물에 데이터 출처를 남긴다
    if os.path.exists(os.path.join(adapter, "adapter_config.json")) and not args.force:
        print(f"[{name}] adapter 존재 → 학습 스킵 (--force로 재학습)")
        return adapter, cfg_path

    cmd = [sys.executable, os.path.join("src", "train.py"), "--config", cfg_path]
    for flag, val in (("--epochs", args.epochs), ("--limit", args.limit),
                      ("--max-steps", args.max_steps), ("--eval-batches", args.eval_batches)):
        if val:
            cmd += [flag, str(val)]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    run(cmd, os.path.join(adir, "train.log"))
    return adapter, cfg_path
```

`eval_arm`을 교체 — **평가에도 같은 config를 넘긴다** (기존 결함 수정):

```python
def eval_arm(name, adapter, cfg_path, args):
    out = eval_json_path(name)
    if os.path.exists(out) and not args.force:
        print(f"[{name}] 평가 결과 존재 → 스킵")
        return
    cmd = [sys.executable, os.path.join("src", "eval_retrieval.py"),
           "--adapter", adapter, "--out", os.path.dirname(out),
           "--config", cfg_path]                      # 학습과 동일 홀드아웃 보장
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    run(cmd, os.path.join(registry.method_dir(name), "eval.log"))
```

`main()`의 arm 검증과 루프를 교체:

```python
    ap.add_argument("--image-root", default="", help="이미지 루트 (train/eval 동일 적용)")
    ...
    names = ["base"] + [a for a in args.arms if a != "base"]
    known = registry.list_methods()
    unknown = [a for a in names if a != "base" and a not in known]
    if unknown:
        sys.exit(f"알 수 없는 메서드: {unknown}  (가능: {known})")

    if args.report_only:
        report(names); return

    for name in names:
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        os.makedirs(registry.method_dir(name), exist_ok=True)
        if name == "base":
            # 튜닝 전 베이스 모델: 학습 없이 평가만. baseline의 resolved를 기준 config로 쓴다
            base_cfg = registry.write_resolved("baseline")
            eval_arm(name, "none", base_cfg, args)
        else:
            adapter, cfg_path = train_arm(name, args)
            eval_arm(name, adapter, cfg_path, args)
    report(names)
```

추가로 `main()`에서 아래 두 가지를 정리한다.

- `ap.add_argument("--config", default="configs/lora_clip.yaml")` 줄을 삭제한다. 베이스 config는 이제 각 메서드 YAML의 `extends`가 지정하므로 러너 인자로 받을 필요가 없다.
- `base_cfg = yaml.safe_load(open(os.path.join(ROOT, args.config), encoding="utf-8"))` 줄을 삭제한다. 병합은 레지스트리가 하므로 러너가 베이스 config를 읽을 이유가 없다. 이에 따라 `import yaml`도 쓰이지 않으면 지운다.

- [ ] **Step 5: 어댑터 저장 경로가 규약과 맞는지 확인**

`src/train.py`가 어댑터를 `{output_dir}/final`에 저장하는지 확인한다:

Run: `grep -n 'final\|output_dir' src/train.py | head -20`
Expected: `os.path.join(t["output_dir"], "final")` 형태가 보인다. 다르면 `train_arm`의 `adapter` 경로를 실제 저장 경로에 맞춘다.

- [ ] **Step 6: 러너 스모크 (학습 없이 config 생성만 확인)**

Run:

```bash
./.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'src')
import registry
for m in registry.list_methods():
    p = registry.write_resolved(m)
    print(m, '->', p)
"
```

Expected: 메서드 6개 각각 `outputs/methods/<name>/config.resolved.yaml` 생성.

- [ ] **Step 7: 전체 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: 전부 PASS

- [ ] **Step 8: 커밋**

```bash
git add configs/methods scripts/run_ablation.py tests/test_method_configs.py
git commit -m "기존 arm을 메서드 YAML로 이관하고 러너를 레지스트리 기반으로 전환

ARMS 딕셔너리가 방법 정의의 두 번째 진실원천이 되어 있던 것을 제거.
러너가 config.resolved.yaml을 만들어 학습과 평가 양쪽에 같은 config를 넘기므로,
평가만 configs/lora_clip.yaml을 읽어 다른 홀드아웃을 보던 결함도 해소된다."
```

---

### Task 5: 로카르노 계층화 부분집합 생성기

**배경:** 472k 도면으로 5개 방법을 학습하는 것은 비현실적이다(방법당 14,769스텝/epoch). 로카르노 분포를 유지한 채 `design_id` 단위로 뽑아 도면 약 100k(디자인 약 12,900개) 표본을 만든다. **design 단위로 뽑아야** 한 디자인의 뷰가 쪼개지지 않아 img2img 학습 신호가 보존된다.

**Files:**
- Create: `scripts/build_subset.py`
- Create: `tests/test_build_subset.py`

**Interfaces:**
- Consumes: `dataset.load_records`
- Produces: `build_subset(records: list[dict], target_drawings: int, seed: int) -> list[dict]` — `scripts/build_subset.py`. 출력은 입력 순서를 유지한 부분 리스트.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_build_subset.py` 생성:

```python
"""부분집합 생성: design 단위 + 로카르노 분포 유지."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_subset import build_subset  # noqa: E402


def _recs():
    """디자인 300개 × 뷰 8장 = 2400 도면. 로카르노 3종을 60/30/10 비율로."""
    out = []
    for d in range(300):
        code = "0101" if d < 180 else ("0202" if d < 270 else "0303")
        for v in range(8):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}",
                        "locarno": code, "text": "t"})
    return out


def test_한_디자인의_뷰는_전부_포함되거나_전부_제외된다():
    got = build_subset(_recs(), target_drawings=800, seed=42)
    per = collections.Counter(r["design_id"] for r in got)
    assert set(per.values()) == {8}, f"뷰가 쪼개진 디자인이 있다: {per}"


def test_목표_도면수에_근접한다():
    got = build_subset(_recs(), target_drawings=800, seed=42)
    assert 720 <= len(got) <= 880, f"목표 800에서 벗어남: {len(got)}"


def test_로카르노_분포가_유지된다():
    recs = _recs()
    got = build_subset(recs, target_drawings=800, seed=42)
    def share(rs):
        c = collections.Counter(r["locarno"] for r in rs)
        return {k: v / len(rs) for k, v in c.items()}
    src, sub = share(recs), share(got)
    for code in src:
        assert abs(src[code] - sub.get(code, 0)) < 0.05, f"{code} 비율이 크게 틀어짐"


def test_같은_seed는_같은_결과():
    a = build_subset(_recs(), 800, seed=42)
    b = build_subset(_recs(), 800, seed=42)
    assert [r["image"] for r in a] == [r["image"] for r in b]


def test_목표가_전체보다_크면_원본_그대로():
    recs = _recs()
    assert len(build_subset(recs, 999999, seed=42)) == len(recs)


def test_입력_순서를_유지한다():
    recs = _recs()
    got = build_subset(recs, 800, seed=42)
    keep = {r["image"] for r in got}
    expected = [r["image"] for r in recs if r["image"] in keep]
    assert [r["image"] for r in got] == expected
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_subset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_subset'`

- [ ] **Step 3: `scripts/build_subset.py` 구현**

```python
"""로카르노 계층화 design 단위 부분집합 생성 (방법 비교 단계용).

전체 472k 도면으로 5개 방법을 학습하는 것은 비현실적이다(방법당 14,769스텝/epoch).
로카르노 코드 비율을 유지한 채 design_id 단위로 뽑아 축소 표본을 만든다.
design 단위인 이유: 한 디자인의 뷰가 쪼개지면 img2img supcon 신호가 깨지고
split_by_design 홀드아웃과도 어긋난다.

  python scripts/build_subset.py --target 100000 --out data/subset_100k.jsonl
"""
import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import load_records  # noqa: E402


def build_subset(records, target_drawings, seed):
    """로카르노 비율을 유지하며 design 단위로 target_drawings장 근처까지 뽑는다.

    반환은 입력 순서를 유지한 부분 리스트 (진단·재현이 쉽도록).
    """
    if target_drawings >= len(records):
        return records

    # design_id -> 도면 수, design_id -> 로카르노(첫 레코드 기준)
    per_design = collections.Counter()
    design_code = {}
    for r in records:
        d = r.get("design_id", r["image"])
        per_design[d] += 1
        design_code.setdefault(d, (r.get("locarno") or "").strip())

    by_code = collections.defaultdict(list)
    for d in per_design:
        by_code[design_code[d]].append(d)

    total = len(records)
    rs = random.Random(seed)
    keep = set()
    for code, designs in sorted(by_code.items()):        # 정렬로 결정성 확보
        rs.shuffle(designs)
        code_drawings = sum(per_design[d] for d in designs)
        quota = target_drawings * code_drawings / total   # 이 코드가 가져갈 도면 수
        got = 0
        for d in designs:
            if got >= quota:
                break
            keep.add(d)
            got += per_design[d]

    return [r for r in records if r.get("design_id", r["image"]) in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pairs.jsonl")
    ap.add_argument("--target", type=int, default=100000, help="목표 도면 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/subset_100k.jsonl")
    args = ap.parse_args()

    records = load_records(os.path.join(ROOT, args.data))
    sub = build_subset(records, args.target, args.seed)
    designs = len({r.get("design_id") for r in sub})
    out = os.path.join(ROOT, args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"subset: {len(sub)} drawings / {designs} designs -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_subset.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 실제 데이터로 생성**

Run: `./.venv/Scripts/python.exe scripts/build_subset.py --target 100000 --out data/subset_100k.jsonl`
Expected: `subset: 약 100000 drawings / 약 12900 designs -> .../data/subset_100k.jsonl`

- [ ] **Step 6: 커밋** (생성된 jsonl은 용량이 크므로 커밋하지 않는다)

```bash
echo "data/subset_*.jsonl" >> .gitignore
git add scripts/build_subset.py tests/test_build_subset.py .gitignore
git commit -m "로카르노 계층화 design 단위 부분집합 생성기 추가

472k 전체로 5개 방법을 학습하는 것은 비현실적이라 비교 단계용 축소 표본을 만든다.
design 단위로 뽑아 한 디자인의 뷰가 쪼개지지 않게 하고, 로카르노 비율을 유지해
축소 표본이 특정 물품군에 치우치지 않도록 한다."
```

---

### Task 6: 인덱스 지문(`index_meta.json`)과 방법별 인덱스 빌드 단계

**배경:** 방법이 5개가 되면 "hobit 인덱스에 tic 어댑터로 검색하는" 사고가 실제로 일어나고, 그건 에러 없이 결과만 조용히 틀어진다. 인덱스에 지문을 남겨 짝이 맞는지 검증 가능하게 한다. `src/embed.py`의 `ckpt_key`와 같은 원칙이다.

**Files:**
- Modify: `src/embed.py` (`build` 분기에 지문 기록, 검증 함수 추가)
- Modify: `scripts/run_ablation.py` (인덱스 빌드 단계 추가)
- Create: `tests/test_index_meta.py`

**Interfaces:**
- Consumes: `registry.method_dir` (Task 3), `train_arm`/`eval_arm` (Task 4)
- Produces:
  - `index_fingerprint(cfg: dict, adapter: str, data: str) -> dict` — `src/embed.py`
  - `check_index_meta(index_dir: str, cfg: dict, adapter: str, data: str) -> tuple[bool, str]` — `(정상여부, 사유)`. Phase 4의 서버가 이 함수로 방법별 활성 여부를 판단한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_index_meta.py` 생성:

```python
"""인덱스 지문: 어댑터·데이터가 뒤바뀐 인덱스를 조용히 쓰지 않는지 고정."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from embed import check_index_meta, index_fingerprint  # noqa: E402

CFG = {"model": {"model_id": "models/base", "image_size": 224}}


def _write(tmp_path, fp):
    d = tmp_path / "index"
    d.mkdir(exist_ok=True)
    (d / "index_meta.json").write_text(json.dumps(fp), encoding="utf-8")
    return str(d)


def test_지문이_같으면_통과(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl")
    assert ok, why


def test_어댑터가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/b", "data/pairs.jsonl")
    assert not ok and "adapter" in why


def test_데이터가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/subset_100k.jsonl")
    assert not ok and "data" in why


def test_해상도가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    other = {"model": {"model_id": "models/base", "image_size": 384}}
    ok, why = check_index_meta(_write(tmp_path, fp), other, "adapters/a", "data/pairs.jsonl")
    assert not ok and "image_size" in why


def test_지문_파일이_없으면_거부(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    ok, why = check_index_meta(str(d), CFG, "adapters/a", "data/pairs.jsonl")
    assert not ok and "index_meta.json" in why
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_index_meta.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_index_meta'`

- [ ] **Step 3: `src/embed.py`에 지문 함수 추가**

`load_tuned` 위에 추가:

```python
def index_fingerprint(cfg, adapter, data):
    """인덱스가 어떤 모델·어댑터·데이터로 만들어졌는지 기록하는 지문.

    방법이 여러 개면 'A 인덱스에 B 어댑터'로 검색하는 사고가 실제로 일어나고,
    그건 에러 없이 결과만 조용히 틀어진다. encode_images의 ckpt_key와 같은 원칙.
    """
    return {
        "model_id": cfg["model"]["model_id"],
        "image_size": cfg["model"]["image_size"],
        "adapter": str(adapter).replace("\\", "/"),
        "data": str(data).replace("\\", "/"),
        "method": cfg.get("method", {}).get("name", ""),
    }


def check_index_meta(index_dir, cfg, adapter, data):
    """인덱스 지문이 현재 설정과 맞는지 확인. (정상여부, 사유) 반환.

    서버는 이 결과로 방법별 활성 여부를 정한다 — 불일치 시 그 방법만 비활성시키고
    서버 전체를 죽이지 않는다.
    """
    p = os.path.join(index_dir, "index_meta.json")
    if not os.path.exists(p):
        return False, f"index_meta.json 없음: {p}"
    saved = json.load(open(p, encoding="utf-8"))
    want = index_fingerprint(cfg, adapter, data)
    for k, v in want.items():
        if k == "method":                      # 이름은 참고용, 판정에 쓰지 않는다
            continue
        if saved.get(k) != v:
            return False, f"{k} 불일치: 인덱스={saved.get(k)!r} 요청={v!r}"
    return True, ""
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_index_meta.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: `build` 분기가 지문을 기록하도록 수정**

`src/embed.py`의 `build` 분기에서 `meta.jsonl` 저장 다음, `shutil.rmtree` 앞에 추가:

```python
        with open(os.path.join(args.index, "index_meta.json"), "w", encoding="utf-8") as f:
            fp = index_fingerprint(cfg, args.adapter, args.data)
            fp.update(n_vectors=int(mat.shape[0]), dim=int(mat.shape[1]))
            json.dump(fp, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 6: 러너에 인덱스 빌드 단계 추가**

`scripts/run_ablation.py`에 함수 추가:

```python
def index_arm(name, adapter, cfg_path, args):
    """방법별 FAISS 인덱스 구축. 웹 비교(Phase 4~5)가 이 산출물을 쓴다."""
    idx_dir = os.path.join(registry.method_dir(name), "index")
    if os.path.exists(os.path.join(idx_dir, "index_meta.json")) and not args.force:
        print(f"[{name}] 인덱스 존재 → 스킵")
        return
    cmd = [sys.executable, os.path.join("src", "embed.py"), "build",
           "--config", cfg_path, "--adapter", adapter, "--index", idx_dir]
    if args.data:
        cmd += ["--data", args.data]
    if args.image_root:
        cmd += ["--image-root", args.image_root]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    run(cmd, os.path.join(registry.method_dir(name), "index.log"))
```

`main()`에 인자를 추가하고:

```python
    ap.add_argument("--data", default="", help="학습·평가·인덱스 공통 데이터 (기본: config)")
    ap.add_argument("--no-index", action="store_true", help="인덱스 빌드 생략")
```

`main()`의 루프에서 평가 다음에 호출:

```python
        else:
            adapter, cfg_path = train_arm(name, args)
            eval_arm(name, adapter, cfg_path, args)
            if not args.no_index:
                index_arm(name, adapter, cfg_path, args)
```

- [ ] **Step 7: 소규모 e2e 스모크**

Run:

```bash
./.venv/Scripts/python.exe scripts/run_ablation.py --arms baseline --limit 200 --epochs 1 --eval-batches 1 --force
```

Expected: `outputs/methods/baseline/` 아래에 `config.resolved.yaml`, `final/`, `eval/final.json`, `index/index_meta.json`이 모두 생긴다. 시간이 오래 걸리면 `--max-steps 5`를 추가한다.

- [ ] **Step 8: 지문 검증이 실제로 동작하는지 확인**

Run:

```bash
./.venv/Scripts/python.exe -c "
import sys, yaml; sys.path.insert(0, 'src')
from embed import check_index_meta
cfg = yaml.safe_load(open('outputs/methods/baseline/config.resolved.yaml', encoding='utf-8'))
print(check_index_meta('outputs/methods/baseline/index', cfg, 'outputs/methods/baseline/final', cfg['train']['data_path']))
print(check_index_meta('outputs/methods/baseline/index', cfg, 'outputs/methods/all/final', cfg['train']['data_path']))
"
```

Expected: 첫 줄 `(True, '')`, 둘째 줄 `(False, "adapter 불일치: ...")`

- [ ] **Step 9: 전체 테스트 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: 전부 PASS

- [ ] **Step 10: 커밋**

```bash
git add src/embed.py scripts/run_ablation.py tests/test_index_meta.py
git commit -m "인덱스 지문 기록·검증과 방법별 인덱스 빌드 단계 추가

방법이 여러 개면 'A 인덱스에 B 어댑터'로 검색하는 사고가 에러 없이 결과만
틀어놓는다. index_meta.json에 모델·해상도·어댑터·데이터를 남기고
check_index_meta()로 검증한다. 러너는 학습→평가→인덱스 3단계를 돈다."
```

---

## 완료 기준

- `./.venv/Scripts/python.exe -m pytest tests/ -v` 전부 통과 (약 40개: 3+4+9+13+6+5)
- `outputs/methods/<name>/`에 `config.resolved.yaml`·`final/`·`eval/final.json`·`index/index_meta.json` 규약이 성립
- `run_ablation.py --arms baseline --limit 200 --max-steps 5`가 학습→평가→인덱스까지 완주
- `data/subset_100k.jsonl` 생성 완료 (도면 약 100k / 디자인 약 12,900)

## 다음 단계

Phase 3(5개 방법 구현)은 방법마다 별도 계획으로 다룬다. `hobit`을 먼저 구현해 이 기반이 실제로 동작하는지 검증하는 것을 권한다. Phase 4~5(서빙·UI)는 방법이 하나만 완성돼도 착수할 수 있으며, `check_index_meta`가 그 진입점이다.
