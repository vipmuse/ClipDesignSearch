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


def test_deep_merge는_over가_건드리지_않은_가지도_base와_공유하지_않는다():
    """over에 없는 중첩 dict가 base와 같은 객체로 남으면, 병합 결과를 나중에
    수정할 때 base가 조용히 오염된다 — resolve()가 cfg["train"]을 덮어쓰는
    패턴이 바로 이 함정을 밟을 수 있는 자리라 별도로 고정한다."""
    base = {"a": {"x": 1}, "b": {"y": 2}}
    got = registry.deep_merge(base, {"a": {"x": 9}})   # over는 "b"를 건드리지 않음
    got["b"]["y"] = 999                                # 병합 결과만 수정
    assert base["b"]["y"] == 2                          # base는 그대로여야 함


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


def test_resolve는_메서드가_지정한_output_dir를_무시하고_규약_경로로_덮어쓴다(
        임시_레지스트리, tmp_path):
    """output_dir는 산출물 경로 규약이라 메서드 YAML이 바꾸지 못해야 한다.
    이 속성이 깨지면(강제 덮어쓰기 줄이 사라지면) 병합의 자연스러운 결과값이
    우연히 맞는 값과 같아지는 일이 없도록, 일부러 다른 값을 넣어 확인한다."""
    (tmp_path / "configs" / "methods" / "rogue.yaml").write_text(
        yaml.safe_dump({
            "name": "rogue",
            "extends": "configs/lora_clip.yaml",
            "train": {"output_dir": "outputs/somewhere/else"},
        }), encoding="utf-8")
    cfg = registry.resolve("rogue")
    assert cfg["train"]["output_dir"] == "outputs/methods/rogue"


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


# ── 최상위 키 검증: 오타가 조용히 '전부 ON' 레시피로 둔갑하는 것을 막는다 ──

def test_모르는_최상위_키는_에러(임시_레지스트리, tmp_path):
    """`train:`을 `trian:`으로 오타내면 병합이 조용히 성공해 그 방법이 베이스(=전부 ON)
    레시피가 된다. ablation은 Δ≈0을 찍고 '어떤 개선도 효과 없다'는 결론이 나온다 —
    몇 시간짜리 학습을 태운 뒤에야 드러나는 사고라 resolve()에서 즉시 막아야 한다."""
    (tmp_path / "configs" / "methods" / "오타.yaml").write_text(
        yaml.safe_dump({
            "name": "오타",
            "extends": "configs/lora_clip.yaml",
            "trian": {"augment": False},          # train 오타
        }, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        registry.resolve("오타")
    assert "trian" in str(e.value)


def test_오타난_블록은_베이스_값으로_조용히_대체되지_않는다(임시_레지스트리, tmp_path):
    """오타 방법이 '베이스와 동일한 설정'으로 resolve되는 일이 없어야 한다는
    본질을 값으로 고정한다 (예외 종류가 아니라 결과를 본다)."""
    (tmp_path / "configs" / "methods" / "오타2.yaml").write_text(
        yaml.safe_dump({"name": "오타2", "extends": "configs/lora_clip.yaml",
                        "trian": {"augment": False}}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(Exception):
        registry.resolve("오타2")


def test_허용된_최상위_키만_쓰면_통과한다(임시_레지스트리, tmp_path):
    (tmp_path / "configs" / "methods" / "full.yaml").write_text(
        yaml.safe_dump({
            "name": "full", "description": "d", "extends": "configs/lora_clip.yaml",
            "data": {"builder": "shared", "pairs": "data/pairs.jsonl"},
            "model": {"image_size": 384}, "lora": {"r": 32}, "train": {"epochs": 2},
        }), encoding="utf-8")
    cfg = registry.resolve("full")
    assert cfg["model"]["image_size"] == 384 and cfg["lora"]["r"] == 32


def test_train_블록이_없으면_파일명이_담긴_에러(임시_레지스트리, tmp_path):
    """베이스에도 메서드에도 train이 없으면 bare KeyError가 아니라 어느 파일이
    문제인지 알려주는 메시지가 나와야 한다."""
    (tmp_path / "configs" / "no_train.yaml").write_text(
        yaml.safe_dump({"model": {"model_id": "m"}, "lora": {}}), encoding="utf-8")
    (tmp_path / "configs" / "methods" / "notrain.yaml").write_text(
        yaml.safe_dump({"name": "notrain", "extends": "configs/no_train.yaml"}),
        encoding="utf-8")
    with pytest.raises(ValueError) as e:
        registry.resolve("notrain")
    assert "notrain.yaml" in str(e.value) and "train" in str(e.value)


# ── data.pairs 배선: YAML의 data 블록이 실제로 학습 데이터를 고른다 ──

def _쓰기(tmp_path, name, spec):
    (tmp_path / "configs" / "methods" / f"{name}.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")


def test_data_pairs가_train_data_path를_결정한다(임시_레지스트리, tmp_path):
    """data 블록이 장식이면 scripts/build_subset.py가 만든 부분집합을 레지스트리에서
    쓸 방법이 없다 — 'YAML 한 장이 방법 하나를 완전히 규정한다'가 깨진다."""
    _쓰기(tmp_path, "sub", {"name": "sub", "extends": "configs/lora_clip.yaml",
                            "data": {"builder": "shared", "pairs": "data/subset_100k.jsonl"}})
    cfg = registry.resolve("sub")
    assert cfg["train"]["data_path"] == "data/subset_100k.jsonl"
    assert cfg["method"]["data"]["pairs"] == "data/subset_100k.jsonl"


def test_포인터_파일도_같은_데이터를_가리킨다(임시_레지스트리, tmp_path):
    import json
    _쓰기(tmp_path, "sub2", {"name": "sub2", "extends": "configs/lora_clip.yaml",
                             "data": {"builder": "shared", "pairs": "data/subset_100k.jsonl"}})
    cfg = registry.resolve("sub2")
    actual = registry.write_data_pointer("sub2", cfg)
    pointer = os.path.join(registry.method_dir("sub2"), "data", "pairs.jsonl.pointer.json")
    assert json.load(open(pointer, encoding="utf-8"))["source"] == "data/subset_100k.jsonl"
    assert actual.replace(os.sep, "/").endswith("data/subset_100k.jsonl")


def test_train_data_path만_고쳐도_포인터가_거짓말하지_않는다(임시_레지스트리, tmp_path):
    """data 블록 없이 train.data_path만 손댄 경우에도 포인터(=출처 기록)가 실제
    학습 데이터와 같아야 한다. 출처를 남기는 것이 유일한 임무인 파일이 틀린 출처를
    남기면 그 파일은 해롭다."""
    import json
    _쓰기(tmp_path, "직접", {"name": "직접", "extends": "configs/lora_clip.yaml",
                             "train": {"data_path": "data/subset_100k.jsonl"}})
    cfg = registry.resolve("직접")
    registry.write_data_pointer("직접", cfg)
    pointer = os.path.join(registry.method_dir("직접"), "data", "pairs.jsonl.pointer.json")
    assert json.load(open(pointer, encoding="utf-8"))["source"] == "data/subset_100k.jsonl"


def test_data_pairs와_train_data_path가_충돌하면_에러(임시_레지스트리, tmp_path):
    """어느 쪽이 이겼는지 산출물만 봐서는 알 수 없으므로 고르지 않고 거부한다."""
    _쓰기(tmp_path, "충돌", {"name": "충돌", "extends": "configs/lora_clip.yaml",
                             "data": {"pairs": "data/a.jsonl"},
                             "train": {"data_path": "data/b.jsonl"}})
    with pytest.raises(ValueError) as e:
        registry.resolve("충돌")
    assert "data/a.jsonl" in str(e.value) and "data/b.jsonl" in str(e.value)


def test_두_곳의_값이_같으면_충돌이_아니다(임시_레지스트리, tmp_path):
    _쓰기(tmp_path, "동일", {"name": "동일", "extends": "configs/lora_clip.yaml",
                             "data": {"pairs": "data/x.jsonl"},
                             "train": {"data_path": "data/x.jsonl"}})
    assert registry.resolve("동일")["train"]["data_path"] == "data/x.jsonl"
