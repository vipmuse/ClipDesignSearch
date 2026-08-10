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
